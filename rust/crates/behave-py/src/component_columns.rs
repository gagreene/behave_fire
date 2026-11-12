//! Direct primitive component buffers; full model batches keep their own owners.
use crate::component_schema::Column;
use behave_fire::engine::FieldKind;
use behave_fire::output::{fill_chunks, ColumnValues, OutputShard};
use behave_fire::ParallelError;

fn initialized<T: Clone>(count: usize, value: T) -> Result<Vec<T>, String> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|e| format!("MemoryError: {e}"))?;
    result.resize(count, value);
    Ok(result)
}

/// One owning primitive column per component result, with no mask/dtype sidecars.
pub(crate) struct Columns {
    pub(crate) columns: Vec<(Column, ColumnValues)>,
    cells: usize,
}
impl Columns {
    /// Allocate final initialized storage from the operation's fixed metadata.
    pub(crate) fn new(schema: Vec<Column>, cells: usize) -> Result<Self, String> {
        let mut columns = Vec::new();
        columns
            .try_reserve_exact(schema.len())
            .map_err(|e| format!("MemoryError: {e}"))?;
        for field in schema {
            let values = match field.kind {
                FieldKind::Physical => ColumnValues::Float64(initialized(cells, f64::NAN)?),
                FieldKind::Code => ColumnValues::Int32(initialized(cells, -1)?),
                FieldKind::Flag => ColumnValues::Bool(initialized(cells, false)?),
            };
            columns.push((field, values));
        }
        Ok(Self { columns, cells })
    }
    /// Fill a single physical result without constructing generic per-row writers.
    pub(crate) fn fill_physical<F>(
        &mut self,
        threads: usize,
        evaluate: F,
    ) -> Result<(), ParallelError<String>>
    where
        F: Fn(usize) -> f64 + Sync,
    {
        let [(_, ColumnValues::Float64(values))] = self.columns.as_mut_slice() else {
            return Err(ParallelError::Work(
                "expected one physical output column".into(),
            ));
        };
        fill_chunks(
            self.cells,
            threads,
            &mut [values.as_mut_slice()],
            &mut [],
            &mut [],
            |range, output| {
                for (local, row) in range.enumerate() {
                    #[cfg(feature = "allocation-probe")]
                    let _scope =
                        crate::allocation_probe::Scope::enter(crate::allocation_probe::KERNEL);
                    let value = evaluate(row);
                    #[cfg(feature = "allocation-probe")]
                    let _encoder =
                        crate::allocation_probe::Scope::enter(crate::allocation_probe::ENCODER);
                    output.physical[0][local] = value;
                }
                Ok(())
            },
        )
    }
    /// Write disjoint final slices; empty grids invoke no scientific callback.
    pub(crate) fn fill<F>(
        &mut self,
        threads: usize,
        evaluate: F,
    ) -> Result<(), ParallelError<String>>
    where
        F: Fn(usize, &mut RowWriter<'_, '_>) -> Result<(), String> + Sync,
    {
        let mut physical = Vec::new();
        let mut codes = Vec::new();
        let mut flags = Vec::new();
        for (_, values) in &mut self.columns {
            match values {
                ColumnValues::Float64(v) => physical.push(v.as_mut_slice()),
                ColumnValues::Int32(v) => codes.push(v.as_mut_slice()),
                ColumnValues::Bool(v) => flags.push(v.as_mut_slice()),
            }
        }
        fill_chunks(
            self.cells,
            threads,
            &mut physical,
            &mut codes,
            &mut flags,
            |range, output| {
                for (local, index) in range.enumerate() {
                    let mut row = RowWriter {
                        output,
                        row: local,
                        physical: 0,
                        codes: 0,
                        flags: 0,
                        failed: false,
                    };
                    #[cfg(feature = "allocation-probe")]
                    let _scope =
                        crate::allocation_probe::Scope::enter(crate::allocation_probe::KERNEL);
                    evaluate(index, &mut row)?;
                    row.finish()?;
                }
                Ok(())
            },
        )
    }
}

/// Ordered scalar writes into final typed planes, without per-cell names or tags.
pub(crate) struct RowWriter<'a, 'b> {
    output: &'a mut OutputShard<'b>,
    row: usize,
    physical: usize,
    codes: usize,
    flags: usize,
    failed: bool,
}
impl RowWriter<'_, '_> {
    /// Store one physical result in the next prepared physical slot.
    pub(crate) fn f64(&mut self, _name: &'static str, value: f64) {
        #[cfg(feature = "allocation-probe")]
        let _scope = crate::allocation_probe::Scope::enter(crate::allocation_probe::ENCODER);
        if let Some(slot) = self
            .output
            .physical
            .get_mut(self.physical)
            .and_then(|v| v.get_mut(self.row))
        {
            *slot = value;
        } else {
            self.failed = true;
        }
        self.physical += 1;
    }
    /// Store an exact classification/status without float64 transit.
    pub(crate) fn i32(&mut self, _name: &'static str, value: i32) {
        #[cfg(feature = "allocation-probe")]
        let _scope = crate::allocation_probe::Scope::enter(crate::allocation_probe::ENCODER);
        if let Some(slot) = self
            .output
            .codes
            .get_mut(self.codes)
            .and_then(|v| v.get_mut(self.row))
        {
            *slot = value;
        } else {
            self.failed = true;
        }
        self.codes += 1;
    }
    /// Store an exact boolean result without a numeric metadata object.
    pub(crate) fn boolean(&mut self, _name: &'static str, value: bool) {
        #[cfg(feature = "allocation-probe")]
        let _scope = crate::allocation_probe::Scope::enter(crate::allocation_probe::ENCODER);
        if let Some(slot) = self
            .output
            .flags
            .get_mut(self.flags)
            .and_then(|v| v.get_mut(self.row))
        {
            *slot = value;
        } else {
            self.failed = true;
        }
        self.flags += 1;
    }
    /// Write a particle component; its full name was resolved in the static schema.
    pub(crate) fn indexed_f64(&mut self, name: &'static str, _index: usize, value: f64) {
        self.f64(name, value);
    }
    /// Visit a nested record whose prefix already belongs to prepared column names.
    pub(crate) fn with_prefix(&mut self, _prefix: &'static str, emit: impl FnOnce(&mut Self)) {
        emit(self);
    }
    /// Reject an incomplete or wrongly typed scientific output sequence.
    fn finish(self) -> Result<(), String> {
        if self.failed
            || self.physical != self.output.physical.len()
            || self.codes != self.output.codes.len()
            || self.flags != self.output.flags.len()
        {
            return Err("component output layout does not match evaluation".into());
        }
        Ok(())
    }
}

//! Typed output buffers and constant metadata for native array results.
use behave_fire::surface::masked::{IntegerKind, MaskedError, MaskedValue, Precision};

pub(crate) fn error_message(error: MaskedError) -> String {
    format!("{}: {error:?}", error.python_exception())
}

fn reserved<T>(count: usize) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|error| format!("MemoryError: {error}"))?;
    Ok(values)
}

fn put<T>(values: &mut Vec<T>, index: usize, value: T, count: usize) -> Result<(), String> {
    if index >= count || index != values.len() {
        return Err(format!(
            "invalid output index {index}; expected {} within {count}",
            values.len()
        ));
    }
    values.push(value);
    Ok(())
}

fn reserve_total<T>(values: &mut Vec<T>, total: usize) -> Result<(), String> {
    values
        .try_reserve_exact(total.saturating_sub(values.len()))
        .map_err(|error| format!("MemoryError: {error}"))
}
fn append_values<T>(values: &mut Vec<T>, mut other: Vec<T>) -> Result<(), String> {
    values
        .try_reserve_exact(other.len())
        .map_err(|error| format!("MemoryError: {error}"))?;
    values.append(&mut other);
    Ok(())
}
fn append_optional<T>(values: &mut Option<Vec<T>>, other: Option<Vec<T>>) -> Result<(), String> {
    match (values, other) {
        (Some(values), Some(other)) => append_values(values, other),
        (None, None) => Ok(()),
        _ => Err("inconsistent output sidecar layout".into()),
    }
}

macro_rules! buffers {
    ($($name:ident: $kind:ty),+ $(,)?) => {
        pub(crate) enum Buffer { $($name(Vec<$kind>)),+ }
        impl Buffer {
            pub(crate) fn len(&self) -> usize { match self { $(Self::$name(v) => v.len()),+ } }
            pub(crate) fn bytes(&self) -> usize { match self { $(Self::$name(v) => v.len() * std::mem::size_of::<$kind>()),+ } }
            fn reserve_total(&mut self, total: usize) -> Result<(), String> { match self { $(Self::$name(v) => reserve_total(v,total)),+ } }
            fn append(&mut self, other: Self) -> Result<(), String> {
                match (self,other) { $((Self::$name(v),Self::$name(other)) => append_values(v,other),)+ _ => Err("inconsistent output buffer type".into()) }
            }
        }
    };
}
buffers! { F64:f64, F32:f32, I8:i8, I16:i16, I32:i32, I64:i64, U8:u8, U16:u16, U32:u32, U64:u64, Bool:bool }

// Captured once per field during setup; boxing would add allocations without shrinking hot-loop state.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy)]
pub(crate) enum SeedValue {
    F64(f64),
    I32(i32),
    Bool(bool),
    Checked(MaskedValue),
}
pub(crate) struct Seed {
    pub name: String,
    pub value: SeedValue,
}
pub(crate) struct DynamicMetadata {
    pub bits: Vec<u64>,
    pub integer_kind: Vec<i8>,
    pub integer_type: Vec<i8>,
    pub single: Vec<bool>,
    pub complex: Vec<bool>,
    pub object_numpy: Vec<bool>,
}
impl DynamicMetadata {
    fn new(count: usize) -> Result<Self, String> {
        Ok(Self {
            bits: reserved(count)?,
            integer_kind: reserved(count)?,
            integer_type: reserved(count)?,
            single: reserved(count)?,
            complex: reserved(count)?,
            object_numpy: reserved(count)?,
        })
    }
}

pub(crate) struct Metadata {
    pub masked: bool,
    pub mask: Option<Vec<bool>>,
    pub single: bool,
    pub complex: bool,
    pub imaginary: Option<Vec<f64>>,
    pub object: bool,
    pub object_numpy: bool,
    pub integer_kind: i8,
    pub integer_type: i8,
    pub integer_any: bool,
    pub integer_consistent: bool,
    pub fill_present: bool,
    pub fill: f64,
    pub dynamic: Option<DynamicMetadata>,
    initial: MaskedValue,
    schema_changed: bool,
}
impl Metadata {
    fn new(value: MaskedValue, count: usize, dynamic: bool) -> Result<Self, String> {
        Ok(Self {
            masked: value.is_masked_array,
            mask: if dynamic || value.is_masked_array {
                Some(reserved(count)?)
            } else {
                None
            },
            single: value.precision == Precision::F32,
            complex: value.complex.is_some(),
            imaginary: if dynamic || value.complex.is_some() {
                Some(reserved(count)?)
            } else {
                None
            },
            object: value.object_result,
            object_numpy: value.object_numpy_scalar,
            integer_kind: value.integer.map_or(-1, |v| v.kind.code() as i8),
            integer_type: value.numpy_integer_type.map_or(-1, |v| v as i8),
            integer_any: value.integer.is_some(),
            integer_consistent: true,
            fill_present: value.fill_value.is_some(),
            fill: value.fill_value.unwrap_or(0.0),
            dynamic: if dynamic {
                Some(DynamicMetadata::new(count)?)
            } else {
                None
            },
            initial: value,
            schema_changed: false,
        })
    }
    fn write(&mut self, index: usize, value: MaskedValue, count: usize) -> Result<(), String> {
        let kind = value.integer.map_or(-1, |v| v.kind.code() as i8);
        let alias = value.numpy_integer_type.map_or(-1, |v| v as i8);
        self.masked |= value.is_masked_array;
        self.single &= value.precision == Precision::F32;
        self.complex |= value.complex.is_some();
        self.object |= value.object_result;
        self.integer_any |= value.integer.is_some();
        self.integer_consistent &= kind == self.integer_kind;
        self.fill_present |= value.fill_value.is_some();
        if let Some(mask) = &mut self.mask {
            put(mask, index, value.mask, count)?;
        }
        if let Some(imaginary) = &mut self.imaginary {
            put(
                imaginary,
                index,
                value.complex.map_or(0.0, |v| v.imag),
                count,
            )?;
        }
        if let Some(dynamic) = &mut self.dynamic {
            put(
                &mut dynamic.bits,
                index,
                value.integer.map_or(0, |v| v.raw_bits),
                count,
            )?;
            put(&mut dynamic.integer_kind, index, kind, count)?;
            put(&mut dynamic.integer_type, index, alias, count)?;
            put(
                &mut dynamic.single,
                index,
                value.precision == Precision::F32,
                count,
            )?;
            put(&mut dynamic.complex, index, value.complex.is_some(), count)?;
            put(
                &mut dynamic.object_numpy,
                index,
                value.object_numpy_scalar,
                count,
            )?;
        } else {
            self.schema_changed |= value.precision != self.initial.precision
                || value.is_masked_array != self.initial.is_masked_array
                || value.object_result != self.initial.object_result
                || value.complex.is_some() != self.initial.complex.is_some()
                || kind != self.integer_kind
                || (value.integer.is_some() && alias != self.integer_type)
                || (value.object_result
                    && value.object_numpy_scalar != self.initial.object_numpy_scalar);
        }
        Ok(())
    }
    fn reserve_total(&mut self, total: usize) -> Result<(), String> {
        if let Some(values) = &mut self.mask {
            reserve_total(values, total)?;
        }
        if let Some(values) = &mut self.imaginary {
            reserve_total(values, total)?;
        }
        if let Some(v) = &mut self.dynamic {
            reserve_total(&mut v.bits, total)?;
            reserve_total(&mut v.integer_kind, total)?;
            reserve_total(&mut v.integer_type, total)?;
            reserve_total(&mut v.single, total)?;
            reserve_total(&mut v.complex, total)?;
            reserve_total(&mut v.object_numpy, total)?;
        }
        Ok(())
    }
    fn append(&mut self, other: Self) -> Result<(), String> {
        self.masked |= other.masked;
        self.single &= other.single;
        self.complex |= other.complex;
        self.object |= other.object;
        self.integer_any |= other.integer_any;
        self.integer_consistent &=
            other.integer_consistent && self.integer_kind == other.integer_kind;
        self.fill_present |= other.fill_present;
        self.schema_changed |= other.schema_changed;
        if self.dynamic.is_none() {
            self.schema_changed |= self.initial.precision != other.initial.precision
                || self.initial.is_masked_array != other.initial.is_masked_array
                || self.initial.object_result != other.initial.object_result
                || self.initial.complex.is_some() != other.initial.complex.is_some()
                || self.integer_kind != other.integer_kind
                || (self.integer_kind >= 0 && self.integer_type != other.integer_type)
                || (self.object && self.object_numpy != other.object_numpy);
        }
        append_optional(&mut self.mask, other.mask)?;
        append_optional(&mut self.imaginary, other.imaginary)?;
        match (&mut self.dynamic, other.dynamic) {
            (Some(v), Some(other)) => {
                append_values(&mut v.bits, other.bits)?;
                append_values(&mut v.integer_kind, other.integer_kind)?;
                append_values(&mut v.integer_type, other.integer_type)?;
                append_values(&mut v.single, other.single)?;
                append_values(&mut v.complex, other.complex)?;
                append_values(&mut v.object_numpy, other.object_numpy)?;
            }
            (None, None) => (),
            _ => return Err("inconsistent dynamic output layout".into()),
        }
        Ok(())
    }
    fn check_lengths(&self, count: usize) -> Result<(), String> {
        if self.mask.as_ref().is_some_and(|v| v.len() != count)
            || self.imaginary.as_ref().is_some_and(|v| v.len() != count)
        {
            return Err("incomplete output sidecar".into());
        }
        if let Some(dynamic) = &self.dynamic {
            if [
                dynamic.bits.len(),
                dynamic.integer_kind.len(),
                dynamic.integer_type.len(),
                dynamic.single.len(),
                dynamic.complex.len(),
                dynamic.object_numpy.len(),
            ]
            .iter()
            .any(|&length| length != count)
            {
                return Err("incomplete dynamic output metadata".into());
            }
        }
        Ok(())
    }
    fn finish(&self, count: usize) -> Result<(), String> {
        self.check_lengths(count)?;
        if self.integer_any
            && !self.object
            && !self.complex
            && (!self.integer_consistent || self.integer_kind < 0)
        {
            return Err("native output has inconsistent integer dtype metadata".into());
        }
        if self.schema_changed {
            return Err("native output schema changed without a dynamic layout".into());
        }
        Ok(())
    }
}

pub(crate) struct Column {
    pub name: String,
    pub data: Buffer,
    pub metadata: Option<Metadata>,
}
pub(crate) struct ColumnBundle {
    pub columns: Vec<Column>,
    pub count: usize,
    first: Vec<SeedValue>,
}
impl ColumnBundle {
    pub(crate) fn from_f64(name: &str, values: Vec<f64>) -> Self {
        let count = values.len();
        Self {
            columns: vec![Column {
                name: name.to_owned(),
                data: Buffer::F64(values),
                metadata: None,
            }],
            count,
            first: Vec::new(),
        }
    }
    pub(crate) fn bytes(&self) -> usize {
        self.columns
            .iter()
            .map(|column| {
                column.data.bytes()
                    + column.metadata.as_ref().map_or(0, |meta| {
                        meta.mask.as_ref().map_or(0, |v| v.len())
                            + meta.imaginary.as_ref().map_or(0, |v| v.len() * 8)
                            + meta.dynamic.as_ref().map_or(0, |v| {
                                v.bits.len() * 8
                                    + v.integer_kind.len()
                                    + v.integer_type.len()
                                    + v.single.len()
                                    + v.complex.len()
                                    + v.object_numpy.len()
                            })
                    })
            })
            .sum()
    }
    pub(crate) fn new(
        seeds: Vec<Seed>,
        count: usize,
        single: bool,
        dynamic: bool,
    ) -> Result<Self, String> {
        for seed in &seeds {
            if let SeedValue::Checked(value) = seed.value {
                if let Some(error) = value.error {
                    return Err(error_message(error));
                }
            }
        }
        let mut columns = reserved(seeds.len())?;
        let mut first = reserved(seeds.len())?;
        for seed in seeds {
            first.push(seed.value);
            let (data, metadata) = match seed.value {
                SeedValue::F64(_) => (
                    if single {
                        Buffer::F32(reserved(count)?)
                    } else {
                        Buffer::F64(reserved(count)?)
                    },
                    None,
                ),
                SeedValue::I32(_) => (Buffer::I32(reserved(count)?), None),
                SeedValue::Bool(_) => (Buffer::Bool(reserved(count)?), None),
                SeedValue::Checked(value) => {
                    value.validate_output().map_err(error_message)?;
                    let dynamic = dynamic || value.object_result;
                    let data = if dynamic || value.object_result || value.complex.is_some() {
                        Buffer::F64(reserved(count)?)
                    } else if let Some(integer) = value.integer {
                        match integer.kind {
                            IntegerKind::I8 => Buffer::I8(reserved(count)?),
                            IntegerKind::I16 => Buffer::I16(reserved(count)?),
                            IntegerKind::I32 => Buffer::I32(reserved(count)?),
                            IntegerKind::I64 => Buffer::I64(reserved(count)?),
                            IntegerKind::U8 => Buffer::U8(reserved(count)?),
                            IntegerKind::U16 => Buffer::U16(reserved(count)?),
                            IntegerKind::U32 => Buffer::U32(reserved(count)?),
                            IntegerKind::U64 => Buffer::U64(reserved(count)?),
                            IntegerKind::Bool => Buffer::Bool(reserved(count)?),
                        }
                    } else if value.precision == Precision::F32 {
                        Buffer::F32(reserved(count)?)
                    } else {
                        Buffer::F64(reserved(count)?)
                    };
                    (data, Some(Metadata::new(value, count, dynamic)?))
                }
            };
            columns.push(Column {
                name: seed.name,
                data,
                metadata,
            });
        }
        Ok(Self {
            columns,
            count,
            first,
        })
    }
    pub(crate) fn reserve_total(&mut self, total: usize) -> Result<(), String> {
        for column in &mut self.columns {
            column.data.reserve_total(total)?;
            if let Some(metadata) = &mut column.metadata {
                metadata.reserve_total(total)?;
            }
        }
        Ok(())
    }
    pub(crate) fn append(&mut self, other: Self) -> Result<(), String> {
        self.check_lengths()?;
        other.check_lengths()?;
        if self.columns.len() != other.columns.len() {
            return Err("inconsistent output field count".into());
        }
        let count = self
            .count
            .checked_add(other.count)
            .ok_or_else(|| "output length overflow".to_owned())?;
        for (column, other) in self.columns.iter_mut().zip(other.columns) {
            if column.name != other.name {
                return Err("inconsistent output field order".into());
            }
            column.data.append(other.data)?;
            match (&mut column.metadata, other.metadata) {
                (Some(metadata), Some(other)) => metadata.append(other)?,
                (None, None) => (),
                _ => return Err("inconsistent output metadata".into()),
            }
        }
        self.count = count;
        Ok(())
    }
    pub(crate) fn check_lengths(&self) -> Result<(), String> {
        for column in &self.columns {
            if column.data.len() != self.count {
                return Err(format!("incomplete output field {}", column.name));
            }
            if let Some(metadata) = &column.metadata {
                metadata.check_lengths(self.count)?;
            }
        }
        Ok(())
    }
    pub(crate) fn write_first(&mut self, measure_encoder: bool) -> Result<(), String> {
        let values = std::mem::take(&mut self.first);
        if self.count == 0 {
            return Ok(());
        }
        if values.len() != self.columns.len() {
            return Err("missing captured output row".into());
        }
        let mut writer = RowWriter::row(self, 0, measure_encoder);
        for value in values {
            match value {
                SeedValue::F64(value) => writer.f64("", value),
                SeedValue::I32(value) => writer.i32("", value),
                SeedValue::Bool(value) => writer.boolean("", value),
                SeedValue::Checked(value) => writer.checked("", value)?,
            }
        }
        writer.finish()
    }
    pub(crate) fn write_f64(
        &mut self,
        field: usize,
        index: usize,
        value: f64,
    ) -> Result<(), String> {
        let column = self
            .columns
            .get_mut(field)
            .ok_or_else(|| "invalid output field".to_owned())?;
        if column.metadata.is_some() {
            return Err("plain write cannot bypass checked metadata".into());
        }
        match &mut column.data {
            Buffer::F64(values) => put(values, index, value, self.count),
            Buffer::F32(values) => put(values, index, value as f32, self.count),
            _ => Err("floating output does not match its declared buffer".into()),
        }
    }
    pub(crate) fn write_i32(
        &mut self,
        field: usize,
        index: usize,
        value: i32,
    ) -> Result<(), String> {
        let column = self
            .columns
            .get_mut(field)
            .ok_or_else(|| "invalid output field".to_owned())?;
        if column.metadata.is_some() {
            return Err("plain write cannot bypass checked metadata".into());
        }
        match &mut column.data {
            Buffer::I32(values) => put(values, index, value, self.count),
            _ => Err("integer classification does not match its declared buffer".into()),
        }
    }
    pub(crate) fn write_bool(
        &mut self,
        field: usize,
        index: usize,
        value: bool,
    ) -> Result<(), String> {
        let column = self
            .columns
            .get_mut(field)
            .ok_or_else(|| "invalid output field".to_owned())?;
        if column.metadata.is_some() {
            return Err("plain write cannot bypass checked metadata".into());
        }
        match &mut column.data {
            Buffer::Bool(values) => put(values, index, value, self.count),
            _ => Err("boolean output does not match its declared buffer".into()),
        }
    }
    pub(crate) fn write_checked(
        &mut self,
        field: usize,
        index: usize,
        value: MaskedValue,
    ) -> Result<(), String> {
        let value = value.validate_output().map_err(error_message)?;
        let column = self
            .columns
            .get_mut(field)
            .ok_or_else(|| "invalid output field".to_owned())?;
        if column.metadata.is_none() {
            return Err("checked output lacks schema".into());
        }
        let bits = value.integer.map_or(0, |v| v.raw_bits);
        match &mut column.data {
            Buffer::F64(values) => put(values, index, value.value, self.count)?,
            Buffer::F32(values) => put(values, index, value.value as f32, self.count)?,
            Buffer::I8(values) => put(values, index, bits as i8, self.count)?,
            Buffer::I16(values) => put(values, index, bits as i16, self.count)?,
            Buffer::I32(values) => put(values, index, bits as i32, self.count)?,
            Buffer::I64(values) => put(values, index, bits as i64, self.count)?,
            Buffer::U8(values) => put(values, index, bits as u8, self.count)?,
            Buffer::U16(values) => put(values, index, bits as u16, self.count)?,
            Buffer::U32(values) => put(values, index, bits as u32, self.count)?,
            Buffer::U64(values) => put(values, index, bits, self.count)?,
            Buffer::Bool(values) => put(values, index, bits != 0, self.count)?,
        }
        column
            .metadata
            .as_mut()
            .ok_or_else(|| "checked output lacks schema".to_owned())?
            .write(index, value, self.count)
    }
    pub(crate) fn finish(&self) -> Result<(), String> {
        for column in &self.columns {
            if column.data.len() != self.count {
                return Err(format!("incomplete output field {}", column.name));
            }
            if let Some(metadata) = &column.metadata {
                metadata.finish(self.count)?;
            }
        }
        Ok(())
    }
}

enum Destination<'a> {
    Record(&'a mut Vec<Seed>),
    Columns {
        bundle: &'a mut ColumnBundle,
        index: usize,
    },
}

/// Emit fields in the operation's fixed order; names are materialized only during setup.
pub(crate) struct RowWriter<'a> {
    destination: Destination<'a>,
    cursor: usize,
    prefix: Option<&'static str>,
    error: Option<String>,
    measure_encoder: bool,
    #[cfg(feature = "allocation-probe")]
    encoder_scope: Option<crate::allocation_probe::Scope>,
}
impl<'a> RowWriter<'a> {
    pub(crate) fn record(seeds: &'a mut Vec<Seed>) -> Self {
        Self {
            destination: Destination::Record(seeds),
            cursor: 0,
            prefix: None,
            error: None,
            measure_encoder: false,
            #[cfg(feature = "allocation-probe")]
            encoder_scope: None,
        }
    }
    pub(crate) fn row(bundle: &'a mut ColumnBundle, index: usize, measure_encoder: bool) -> Self {
        Self {
            destination: Destination::Columns { bundle, index },
            cursor: 0,
            prefix: None,
            error: None,
            measure_encoder,
            #[cfg(feature = "allocation-probe")]
            encoder_scope: None,
        }
    }
    fn name(prefix: Option<&str>, name: &str, index: Option<usize>) -> String {
        match (prefix, index) {
            (Some(prefix), Some(index)) => format!("{prefix}.{name}.{index}"),
            (Some(prefix), None) => format!("{prefix}.{name}"),
            (None, Some(index)) => format!("{name}.{index}"),
            (None, None) => name.to_owned(),
        }
    }
    fn start_encoding(&mut self) {
        #[cfg(feature = "allocation-probe")]
        if self.measure_encoder && self.encoder_scope.is_none() {
            self.encoder_scope = Some(crate::allocation_probe::Scope::enter(
                crate::allocation_probe::ENCODER,
            ));
        }
        #[cfg(not(feature = "allocation-probe"))]
        let _ = self.measure_encoder;
    }
    fn remember_error(&mut self, result: Result<(), String>) {
        if let Err(error) = result {
            if self.error.is_none() {
                self.error = Some(error);
            }
        }
        self.cursor += 1;
    }
    pub(crate) fn f64(&mut self, name: &'static str, value: f64) {
        self.indexed(name, None, value);
    }
    pub(crate) fn indexed_f64(&mut self, name: &'static str, index: usize, value: f64) {
        self.indexed(name, Some(index), value);
    }
    fn indexed(&mut self, name: &'static str, component: Option<usize>, value: f64) {
        self.start_encoding();
        let result = match &mut self.destination {
            Destination::Record(seeds) => seeds
                .try_reserve(1)
                .map_err(|error| format!("MemoryError: {error}"))
                .map(|()| {
                    seeds.push(Seed {
                        name: Self::name(self.prefix, name, component),
                        value: SeedValue::F64(value),
                    })
                }),
            Destination::Columns { bundle, index } => bundle.write_f64(self.cursor, *index, value),
        };
        self.remember_error(result);
    }
    pub(crate) fn i32(&mut self, name: &'static str, value: i32) {
        self.start_encoding();
        let result = match &mut self.destination {
            Destination::Record(seeds) => seeds
                .try_reserve(1)
                .map_err(|error| format!("MemoryError: {error}"))
                .map(|()| {
                    seeds.push(Seed {
                        name: Self::name(self.prefix, name, None),
                        value: SeedValue::I32(value),
                    })
                }),
            Destination::Columns { bundle, index } => bundle.write_i32(self.cursor, *index, value),
        };
        self.remember_error(result);
    }
    pub(crate) fn boolean(&mut self, name: &'static str, value: bool) {
        self.start_encoding();
        let result = match &mut self.destination {
            Destination::Record(seeds) => seeds
                .try_reserve(1)
                .map_err(|error| format!("MemoryError: {error}"))
                .map(|()| {
                    seeds.push(Seed {
                        name: Self::name(self.prefix, name, None),
                        value: SeedValue::Bool(value),
                    })
                }),
            Destination::Columns { bundle, index } => bundle.write_bool(self.cursor, *index, value),
        };
        self.remember_error(result);
    }
    pub(crate) fn checked(&mut self, name: &'static str, value: MaskedValue) -> Result<(), String> {
        let value = value.validate_output().map_err(error_message)?;
        self.start_encoding();
        let result = match &mut self.destination {
            Destination::Record(seeds) => seeds
                .try_reserve(1)
                .map_err(|error| format!("MemoryError: {error}"))
                .map(|()| {
                    seeds.push(Seed {
                        name: Self::name(self.prefix, name, None),
                        value: SeedValue::Checked(value),
                    })
                }),
            Destination::Columns { bundle, index } => {
                bundle.write_checked(self.cursor, *index, value)
            }
        };
        self.cursor += 1;
        if let Err(error) = &result {
            if self.error.is_none() {
                self.error = Some(error.clone());
            }
        }
        result
    }
    pub(crate) fn with_prefix<T>(
        &mut self,
        prefix: &'static str,
        run: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let previous = self.prefix.replace(prefix);
        let result = run(self);
        self.prefix = previous;
        result
    }
    pub(crate) fn finish(mut self) -> Result<(), String> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        match &self.destination {
            Destination::Record(seeds) => {
                for (index, seed) in seeds.iter().enumerate() {
                    if seeds[..index].iter().any(|other| other.name == seed.name) {
                        return Err(format!("duplicate output field {}", seed.name));
                    }
                }
            }
            Destination::Columns { bundle, .. } => {
                if self.cursor != bundle.columns.len() {
                    return Err("incomplete output row".into());
                }
            }
        }
        Ok(())
    }
}
#[cfg(all(test, feature = "allocation-probe"))]
mod test_support {
    use super::*;
    pub(super) fn seed_f64(name: &str, value: f64) -> Seed {
        Seed {
            name: name.to_owned(),
            value: SeedValue::F64(value),
        }
    }
    pub(super) fn seed_i32(name: &str, value: i32) -> Seed {
        Seed {
            name: name.to_owned(),
            value: SeedValue::I32(value),
        }
    }
    pub(super) fn seed_checked(name: &str, value: MaskedValue) -> Seed {
        Seed {
            name: name.to_owned(),
            value: SeedValue::Checked(value),
        }
    }
}
#[cfg(all(test, feature = "allocation-probe"))]
use test_support::{seed_checked, seed_f64, seed_i32};
#[cfg(all(test, feature = "allocation-probe"))]
mod tests {
    use super::*;
    use behave_fire::surface::masked::{IntegerKind, IntegerValue, MaskedValue, Precision};

    #[test]
    fn checked_columns_preserve_raw_masked_values_and_fill_without_loop_allocations() {
        let seed = MaskedValue::masked_f32(-0.0, false).with_fill_value(Some(-17.0));
        let mut bundle =
            ColumnBundle::new(vec![seed_checked("value", seed)], 1000, false, false).unwrap();
        crate::allocation_probe::start();
        for index in 0..1000 {
            let value =
                MaskedValue::masked_f32(if index % 2 == 0 { -0.0 } else { 3.5 }, index % 3 == 0)
                    .with_fill_value(Some(-17.0));
            bundle.write_checked(0, index, value).unwrap();
        }
        let counters = crate::allocation_probe::stop();
        assert_eq!(counters[0].0, 0);
        assert!(matches!(bundle.columns[0].data, Buffer::F32(_)));
        let Buffer::F32(values) = &bundle.columns[0].data else {
            unreachable!()
        };
        assert_eq!(values[0].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(values[1], 3.5);
        let metadata = bundle.columns[0].metadata.as_ref().unwrap();
        assert_eq!(metadata.fill, -17.0);
        assert!(metadata.fill_present && metadata.masked);
        assert_eq!(
            metadata.mask.as_ref().unwrap()[0..4],
            [true, false, false, true]
        );
    }

    #[test]
    fn exact_integer_storage_and_i32_classification_never_round_through_f64() {
        let value = MaskedValue::plain(0.0)
            .with_integer(IntegerValue::from_bits(IntegerKind::U64, u64::MAX));
        let mut bundle = ColumnBundle::new(
            vec![seed_checked("exact", value), seed_i32("fire_type", 0)],
            2,
            false,
            false,
        )
        .unwrap();
        bundle.write_checked(0, 0, value).unwrap();
        bundle.write_i32(1, 0, 3).unwrap();
        let Buffer::U64(values) = &bundle.columns[0].data else {
            panic!("u64 data was widened")
        };
        assert_eq!(values[0], u64::MAX);
        let Buffer::I32(classes) = &bundle.columns[1].data else {
            panic!("classification was widened")
        };
        assert_eq!(classes[0], 3);
    }

    #[test]
    fn dynamic_payload_metadata_uses_preallocated_sidecars() {
        let first = MaskedValue::plain(1.25).with_object_payload(false);
        let second = MaskedValue::plain_f32(2.5).with_object_payload(true);
        let mut bundle =
            ColumnBundle::new(vec![seed_checked("value", first)], 2, false, true).unwrap();
        crate::allocation_probe::start();
        bundle.write_checked(0, 0, first).unwrap();
        bundle.write_checked(0, 1, second).unwrap();
        let counters = crate::allocation_probe::stop();
        assert_eq!(counters[0].0, 0);
        let metadata = bundle.columns[0].metadata.as_ref().unwrap();
        assert!(metadata.object);
        let sidecars = metadata.dynamic.as_ref().unwrap();
        assert_eq!(sidecars.single, [false, true]);
        assert_eq!(sidecars.object_numpy, [false, true]);
        assert_eq!(second.precision, Precision::F32);
    }

    #[test]
    fn empty_columns_retain_masked_class_and_shape_metadata() {
        let mut bundle = ColumnBundle::new(
            vec![seed_checked("value", MaskedValue::masked(0.0, false))],
            0,
            false,
            false,
        )
        .unwrap();
        assert_eq!(bundle.columns[0].data.len(), 0);
        assert!(bundle.columns[0].metadata.as_ref().unwrap().masked);
        assert!(bundle.write_f64(0, 0, 1.0).is_err());
    }
}

#[cfg(all(test, feature = "allocation-probe"))]
mod integrity_tests {
    use super::*;
    use behave_fire::surface::masked::{IntegerKind, IntegerValue};
    #[test]
    fn plain_writes_cannot_bypass_checked_metadata() {
        let mut bundle = ColumnBundle::new(
            vec![seed_checked("value", MaskedValue::masked(1.0, false))],
            1,
            false,
            false,
        )
        .unwrap();
        assert!(bundle.write_f64(0, 0, 1.0).is_err());
        assert_eq!(bundle.columns[0].data.len(), 0);
        assert!(bundle.finish().is_err());
    }
    #[test]
    fn integers_beyond_float_precision_and_all_false_masks_are_retained() {
        let exact = (1_u64 << 63) + 7;
        let value = MaskedValue::masked(0.0, false)
            .with_integer(IntegerValue::from_bits(IntegerKind::U64, exact));
        let mut bundle =
            ColumnBundle::new(vec![seed_checked("value", value)], 2, false, false).unwrap();
        bundle.write_checked(0, 0, value).unwrap();
        bundle
            .write_checked(
                0,
                1,
                value.with_integer(IntegerValue::from_bits(IntegerKind::U64, u64::MAX)),
            )
            .unwrap();
        bundle.finish().unwrap();
        let Buffer::U64(values) = &bundle.columns[0].data else {
            unreachable!()
        };
        assert_eq!(values, &[exact, u64::MAX]);
        let metadata = bundle.columns[0].metadata.as_ref().unwrap();
        assert!(metadata.masked);
        assert_eq!(metadata.mask.as_ref().unwrap(), &[false, false]);
    }
}

#[cfg(all(test, feature = "allocation-probe"))]
mod row_tests {
    use super::*;
    #[test]
    fn schema_capture_and_row_writes_share_field_order_without_loop_allocations() {
        let mut seeds = Vec::new();
        let mut capture = RowWriter::record(&mut seeds);
        capture.f64("plain", 1.0);
        capture.with_prefix("particles", |out| out.indexed_f64("load", 2, 3.0));
        capture
            .checked("masked", MaskedValue::masked(2.0, false))
            .unwrap();
        capture.finish().unwrap();
        assert_eq!(
            seeds.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            ["plain", "particles.load.2", "masked"]
        );
        let mut bundle = ColumnBundle::new(seeds, 1000, false, false).unwrap();
        bundle.write_first(false).unwrap();
        crate::allocation_probe::start();
        for index in 1..1000 {
            let mut writer = RowWriter::row(&mut bundle, index, false);
            writer.f64("plain", index as f64);
            writer.with_prefix("particles", |out| out.indexed_f64("load", 2, 3.0));
            writer
                .checked("masked", MaskedValue::masked(2.0, index % 2 == 0))
                .unwrap();
            writer.finish().unwrap();
        }
        let counters = crate::allocation_probe::stop();
        assert_eq!(counters[0].0, 0);
        bundle.finish().unwrap();
    }
    #[test]
    fn missing_or_duplicate_fields_fail_instead_of_returning_partial_outputs() {
        let mut seeds = Vec::new();
        let mut capture = RowWriter::record(&mut seeds);
        capture.f64("same", 1.0);
        capture.f64("same", 2.0);
        assert!(capture.finish().is_err());
        let mut bundle = ColumnBundle::new(
            vec![seed_f64("a", 1.0), seed_f64("b", 2.0)],
            1,
            false,
            false,
        )
        .unwrap();
        let mut writer = RowWriter::row(&mut bundle, 0, false);
        writer.f64("a", 1.0);
        assert!(writer.finish().is_err());
    }
}

#[cfg(all(test, feature = "allocation-probe"))]
mod merge_tests {
    use super::*;
    use behave_fire::surface::masked::{IntegerKind, IntegerValue};
    use std::hint::black_box;
    #[test]
    fn encoder_scope_counts_rows_and_excludes_input_preparation() {
        let mut bundle = ColumnBundle::new(vec![seed_f64("value", 1.0)], 4, false, false).unwrap();
        crate::allocation_probe::start();
        bundle.write_first(true).unwrap();
        for index in 1..4 {
            let mut row = RowWriter::row(&mut bundle, index, true);
            black_box(vec![0_u8; 13]);
            row.f64("value", index as f64);
            row.finish().unwrap();
        }
        let counters = crate::allocation_probe::stop();
        assert_eq!(counters[0].0, 3);
        assert_eq!(counters[2], (0, 0, 4, 1));
    }
    #[test]
    fn eager_errors_precede_output_only_integer_restrictions() {
        let mut unsupported = MaskedValue::python_int(1);
        unsupported.integer = None;
        assert!(unsupported.validate_output().is_err());
        let mut eager = MaskedValue::plain(0.0);
        eager.error = Some(MaskedError::PythonZeroDivision);
        let error = ColumnBundle::new(
            vec![
                seed_checked("first", unsupported),
                seed_checked("later", eager),
            ],
            1,
            false,
            true,
        )
        .err()
        .unwrap();
        assert_eq!(error, "ZeroDivisionError: PythonZeroDivision");
    }
    #[test]
    fn chunk_merge_keeps_dynamic_types_bits_and_metadata_reductions() {
        let a = MaskedValue::plain_f32(1.25).with_object_payload(true);
        let b = MaskedValue::plain(0.0)
            .with_integer(IntegerValue::from_bits(IntegerKind::U64, (1_u64 << 63) + 7))
            .with_object_payload(false);
        let mut first = ColumnBundle::new(vec![seed_checked("value", a)], 1, false, true).unwrap();
        let mut second = ColumnBundle::new(vec![seed_checked("value", b)], 1, false, true).unwrap();
        first.write_first(false).unwrap();
        second.write_first(false).unwrap();
        first.reserve_total(2).unwrap();
        first.append(second).unwrap();
        first.finish().unwrap();
        assert_eq!(first.count, 2);
        let metadata = first.columns[0].metadata.as_ref().unwrap();
        assert!(metadata.object && !metadata.single);
        let dynamic = metadata.dynamic.as_ref().unwrap();
        assert_eq!(dynamic.single, [true, false]);
        assert_eq!(dynamic.bits, [0, (1_u64 << 63) + 7]);
        assert_eq!(dynamic.object_numpy, [true, false]);
    }
}

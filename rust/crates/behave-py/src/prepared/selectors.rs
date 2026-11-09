//! Decode integral unit controls once, before any component worker starts.
use behave_fire::units::InvalidUnit;

pub(crate) enum Selected<T> {
    Uniform(T),
    Cells(Vec<T>),
}
impl<T: Copy + TryFrom<i32, Error = InvalidUnit>> Selected<T> {
    pub(crate) fn new(values: &[i64]) -> Result<Self, String> {
        let decode = |&value: &i64| {
            let code =
                i32::try_from(value).map_err(|_| "unit selector exceeds int32".to_owned())?;
            T::try_from(code).map_err(|e| e.to_string())
        };
        if let [value] = values {
            Ok(Self::Uniform(decode(value)?))
        } else {
            values
                .iter()
                .map(decode)
                .collect::<Result<Vec<_>, _>>()
                .map(Self::Cells)
        }
    }
    pub(crate) fn at(&self, row: usize) -> T {
        match self {
            Self::Uniform(value) => *value,
            Self::Cells(values) => values[row],
        }
    }
}

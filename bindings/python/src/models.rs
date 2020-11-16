use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::token::PyToken;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Serialize};
use tk::models::bpe::{BpeBuilder, Merges, Vocab, BPE};
use tk::models::unigram::Unigram;
use tk::models::wordlevel::WordLevel;
use tk::models::wordpiece::{WordPiece, WordPieceBuilder};
use tk::models::ModelWrapper;
use tk::{Model, Token};
use tokenizers as tk;

use super::error::{deprecation_warning, ToPyResult};

/// A Model represents some tokenization algorithm like BPE or Word
/// This class cannot be constructed directly. Please use one of the concrete models.
#[pyclass(module = "tokenizers.models", name=Model)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyModel {
    #[serde(flatten)]
    pub model: Arc<RwLock<ModelWrapper>>,
}

impl PyModel {
    pub(crate) fn get_as_subtype(&self) -> PyResult<PyObject> {
        let base = self.clone();
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(match *self.model.as_ref().read().unwrap() {
            ModelWrapper::BPE(_) => Py::new(py, (PyBPE {}, base))?.into_py(py),
            ModelWrapper::WordPiece(_) => Py::new(py, (PyWordPiece {}, base))?.into_py(py),
            ModelWrapper::WordLevel(_) => Py::new(py, (PyWordLevel {}, base))?.into_py(py),
            ModelWrapper::Unigram(_) => Py::new(py, (PyUnigram {}, base))?.into_py(py),
        })
    }
}

impl Model for PyModel {
    fn tokenize(&self, tokens: &str) -> tk::Result<Vec<Token>> {
        self.model.read().unwrap().tokenize(tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.read().unwrap().token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.read().unwrap().id_to_token(id)
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.model.read().unwrap().get_vocab()
    }

    fn get_vocab_size(&self) -> usize {
        self.model.read().unwrap().get_vocab_size()
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> tk::Result<Vec<PathBuf>> {
        self.model.read().unwrap().save(folder, name)
    }
}

impl<I> From<I> for PyModel
where
    I: Into<ModelWrapper>,
{
    fn from(model: I) -> Self {
        Self {
            model: Arc::new(RwLock::new(model.into())),
        }
    }
}

#[pymethods]
impl PyModel {
    #[new]
    fn __new__() -> PyResult<Self> {
        // Instantiate a default empty model. This doesn't really make sense, but we need
        // to be able to instantiate an empty model for pickle capabilities.
        Ok(PyModel {
            model: Arc::new(RwLock::new(BPE::default().into())),
        })
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.model).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to pickle Model: {}",
                e.to_string()
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.model = serde_json::from_slice(s.as_bytes()).map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle Model: {}",
                        e.to_string()
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn tokenize(&self, tokens: &str) -> PyResult<Vec<PyToken>> {
        Ok(ToPyResult(self.model.read().unwrap().tokenize(tokens))
            .into_py()?
            .into_iter()
            .map(|t| t.into())
            .collect())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.read().unwrap().token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.read().unwrap().id_to_token(id)
    }

    fn save(&self, folder: &str, name: Option<&str>) -> PyResult<Vec<String>> {
        let saved: PyResult<Vec<_>> =
            ToPyResult(self.model.read().unwrap().save(Path::new(folder), name)).into();

        Ok(saved?
            .into_iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect())
    }
}

/// BPE Model
/// Allows the creation of a BPE Model to be used with a Tokenizer
#[pyclass(extends=PyModel, module = "tokenizers.models", name=BPE)]
pub struct PyBPE {}

impl PyBPE {
    fn with_builder(mut builder: BpeBuilder, kwargs: Option<&PyDict>) -> PyResult<(Self, PyModel)> {
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "cache_capacity" => builder = builder.cache_capacity(value.extract()?),
                    "dropout" => {
                        if let Some(dropout) = value.extract()? {
                            builder = builder.dropout(dropout);
                        }
                    }
                    "unk_token" => {
                        if let Some(unk) = value.extract()? {
                            builder = builder.unk_token(unk);
                        }
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(value.extract()?)
                    }
                    "end_of_word_suffix" => builder = builder.end_of_word_suffix(value.extract()?),
                    "fuse_unk" => builder = builder.fuse_unk(value.extract()?),
                    _ => println!("Ignored unknown kwarg option {}", key),
                };
            }
        }

        match builder.build() {
            Err(e) => Err(exceptions::PyException::new_err(format!(
                "Error while initializing BPE: {}",
                e
            ))),
            Ok(bpe) => Ok((PyBPE {}, bpe.into())),
        }
    }
}

#[derive(FromPyObject)]
enum PyVocab<'a> {
    Vocab(Vocab),
    Filename(&'a str),
}
#[derive(FromPyObject)]
enum PyMerges<'a> {
    Merges(Merges),
    Filename(&'a str),
}

#[pymethods]
impl PyBPE {
    #[getter]
    fn get_dropout(self_: PyRef<Self>) -> Option<f64> {
        let super_ = self_.as_ref();
        let model = super_.model.read().unwrap();
        if let ModelWrapper::BPE(ref bpe) = *model {
            bpe.dropout.map(|d| d as f64)
        } else {
            unreachable!()
        }
    }

    #[setter]
    fn set_dropout(self_: PyRef<Self>, dropout: Option<f32>) {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().unwrap();
        if let ModelWrapper::BPE(ref mut bpe) = *model {
            bpe.dropout = dropout;
        }
    }

    #[getter]
    fn get_unk_token(self_: PyRef<Self>) -> Option<String> {
        let super_ = self_.as_ref();
        let model = super_.model.read().unwrap();
        if let ModelWrapper::BPE(ref bpe) = *model {
            bpe.unk_token.clone()
        } else {
            unreachable!()
        }
    }

    #[setter]
    fn set_unk_token(self_: PyRef<Self>, unk_token: Option<String>) {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().unwrap();
        if let ModelWrapper::BPE(ref mut bpe) = *model {
            bpe.unk_token = unk_token;
        }
    }

    #[getter]
    fn get_continuing_subword_prefix(self_: PyRef<Self>) -> Option<String> {
        let super_ = self_.as_ref();
        let model = super_.model.read().unwrap();
        if let ModelWrapper::BPE(ref bpe) = *model {
            bpe.continuing_subword_prefix.clone()
        } else {
            unreachable!()
        }
    }

    #[setter]
    fn set_continuing_subword_prefix(
        self_: PyRef<Self>,
        continuing_subword_prefix: Option<String>,
    ) {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().unwrap();
        if let ModelWrapper::BPE(ref mut bpe) = *model {
            bpe.continuing_subword_prefix = continuing_subword_prefix;
        }
    }

    #[getter]
    fn get_end_of_word_suffix(self_: PyRef<Self>) -> Option<String> {
        let super_ = self_.as_ref();
        let model = super_.model.read().unwrap();
        if let ModelWrapper::BPE(ref bpe) = *model {
            bpe.end_of_word_suffix.clone()
        } else {
            unreachable!()
        }
    }

    #[setter]
    fn set_end_of_word_suffix(self_: PyRef<Self>, end_of_word_suffix: Option<String>) {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().unwrap();
        if let ModelWrapper::BPE(ref mut bpe) = *model {
            bpe.end_of_word_suffix = end_of_word_suffix;
        }
    }

    #[getter]
    fn get_fuse_unk(self_: PyRef<Self>) -> bool {
        let super_ = self_.as_ref();
        let model = super_.model.read().unwrap();
        if let ModelWrapper::BPE(ref bpe) = *model {
            bpe.fuse_unk
        } else {
            unreachable!()
        }
    }

    #[setter]
    fn set_fuse_unk(self_: PyRef<Self>, fuse_unk: bool) {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().unwrap();
        if let ModelWrapper::BPE(ref mut bpe) = *model {
            bpe.fuse_unk = fuse_unk;
        }
    }

    #[new]
    #[args(kwargs = "**")]
    fn new(
        vocab: Option<PyVocab>,
        merges: Option<PyMerges>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<(Self, PyModel)> {
        if (vocab.is_some() && merges.is_none()) || (vocab.is_none() && merges.is_some()) {
            return Err(exceptions::PyValueError::new_err(
                "`vocab` and `merges` must be both specified",
            ));
        }

        let mut builder = BPE::builder();
        if let (Some(vocab), Some(merges)) = (vocab, merges) {
            match (vocab, merges) {
                (PyVocab::Vocab(vocab), PyMerges::Merges(merges)) => {
                    builder = builder.vocab_and_merges(vocab, merges);
                }
                (PyVocab::Filename(vocab_filename), PyMerges::Filename(merges_filename)) => {
                    deprecation_warning(
                    "0.9.0",
                    "BPE.__init__ will not create from files anymore, try `BPE.from_file` instead",
                )?;
                    builder =
                        builder.files(vocab_filename.to_string(), merges_filename.to_string());
                }
                _ => {
                    return Err(exceptions::PyValueError::new_err(
                        "`vocab` and `merges` must be both be from memory or both filenames",
                    ));
                }
            }
        }

        PyBPE::with_builder(builder, kwargs)
    }

    #[staticmethod]
    fn read_file(vocab_filename: &str, merges_filename: &str) -> PyResult<(Vocab, Merges)> {
        BPE::read_file(vocab_filename, merges_filename).map_err(|e| {
            exceptions::PyValueError::new_err(format!(
                "Error while reading vocab&merges files: {}",
                e
            ))
        })
    }

    #[staticmethod]
    #[args(kwargs = "**")]
    fn from_file(
        py: Python,
        vocab_filename: &str,
        merges_filename: &str,
        kwargs: Option<&PyDict>,
    ) -> PyResult<Py<Self>> {
        let (vocab, merges) = BPE::read_file(vocab_filename, merges_filename).map_err(|e| {
            exceptions::PyValueError::new_err(format!("Error while reading BPE files: {}", e))
        })?;
        Py::new(
            py,
            PyBPE::new(
                Some(PyVocab::Vocab(vocab)),
                Some(PyMerges::Merges(merges)),
                kwargs,
            )?,
        )
    }
}

/// WordPiece Model
#[pyclass(extends=PyModel, module = "tokenizers.models", name=WordPiece)]
pub struct PyWordPiece {}

impl PyWordPiece {
    fn with_builder(
        mut builder: WordPieceBuilder,
        kwargs: Option<&PyDict>,
    ) -> PyResult<(Self, PyModel)> {
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "unk_token" => {
                        builder = builder.unk_token(val.extract()?);
                    }
                    "max_input_chars_per_word" => {
                        builder = builder.max_input_chars_per_word(val.extract()?);
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(val.extract()?);
                    }
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }

        match builder.build() {
            Err(e) => Err(exceptions::PyException::new_err(format!(
                "Error while initializing WordPiece: {}",
                e
            ))),
            Ok(wordpiece) => Ok((PyWordPiece {}, wordpiece.into())),
        }
    }
}

#[pymethods]
impl PyWordPiece {
    #[getter]
    fn get_unk_token(self_: PyRef<Self>) -> String {
        let super_ = self_.as_ref();
        let model = super_.model.read().unwrap();
        if let ModelWrapper::WordPiece(ref wp) = *model {
            wp.unk_token.clone()
        } else {
            unreachable!()
        }
    }

    #[setter]
    fn set_unk_token(self_: PyRef<Self>, unk_token: String) {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().unwrap();
        if let ModelWrapper::WordPiece(ref mut wp) = *model {
            wp.unk_token = unk_token;
        }
    }

    #[getter]
    fn get_continuing_subword_prefix(self_: PyRef<Self>) -> String {
        let super_ = self_.as_ref();
        let model = super_.model.read().unwrap();
        if let ModelWrapper::WordPiece(ref wp) = *model {
            wp.continuing_subword_prefix.clone()
        } else {
            unreachable!()
        }
    }

    #[setter]
    fn set_continuing_subword_prefix(self_: PyRef<Self>, continuing_subword_prefix: String) {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().unwrap();
        if let ModelWrapper::WordPiece(ref mut wp) = *model {
            wp.continuing_subword_prefix = continuing_subword_prefix;
        }
    }

    #[getter]
    fn get_max_input_chars_per_word(self_: PyRef<Self>) -> usize {
        let super_ = self_.as_ref();
        let model = super_.model.read().unwrap();
        if let ModelWrapper::WordPiece(ref wp) = *model {
            wp.max_input_chars_per_word
        } else {
            unreachable!()
        }
    }

    #[setter]
    fn set_max_input_chars_per_word(self_: PyRef<Self>, max: usize) {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().unwrap();
        if let ModelWrapper::WordPiece(ref mut wp) = *model {
            wp.max_input_chars_per_word = max;
        }
    }

    #[new]
    #[args(kwargs = "**")]
    fn new(vocab: Option<PyVocab>, kwargs: Option<&PyDict>) -> PyResult<(Self, PyModel)> {
        let mut builder = WordPiece::builder();

        if let Some(vocab) = vocab {
            match vocab {
                PyVocab::Vocab(vocab) => {
                    builder = builder.vocab(vocab);
                }
                PyVocab::Filename(vocab_filename) => {
                    deprecation_warning(
                        "0.9.0",
                        "WordPiece.__init__ will not create from files anymore, try `WordPiece.from_file` instead",
                    )?;
                    builder = builder.files(vocab_filename.to_string());
                }
            }
        }

        PyWordPiece::with_builder(builder, kwargs)
    }

    #[staticmethod]
    fn read_file(vocab_filename: &str) -> PyResult<Vocab> {
        WordPiece::read_file(vocab_filename).map_err(|e| {
            exceptions::PyValueError::new_err(format!("Error while reading WordPiece file: {}", e))
        })
    }

    #[staticmethod]
    #[args(kwargs = "**")]
    fn from_file(py: Python, vocab_filename: &str, kwargs: Option<&PyDict>) -> PyResult<Py<Self>> {
        let vocab = WordPiece::read_file(vocab_filename).map_err(|e| {
            exceptions::PyValueError::new_err(format!("Error while reading WordPiece file: {}", e))
        })?;
        Py::new(py, PyWordPiece::new(Some(PyVocab::Vocab(vocab)), kwargs)?)
    }
}

#[pyclass(extends=PyModel, module = "tokenizers.models", name=WordLevel)]
pub struct PyWordLevel {}

#[pymethods]
impl PyWordLevel {
    #[getter]
    fn get_unk_token(self_: PyRef<Self>) -> String {
        let super_ = self_.as_ref();
        let model = super_.model.read().unwrap();
        if let ModelWrapper::WordLevel(ref wl) = *model {
            wl.unk_token.clone()
        } else {
            unreachable!()
        }
    }

    #[setter]
    fn set_unk_token(self_: PyRef<Self>, unk_token: String) {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().unwrap();
        if let ModelWrapper::WordLevel(ref mut wl) = *model {
            wl.unk_token = unk_token;
        }
    }

    #[new]
    #[args(unk_token = "None")]
    fn new(vocab: Option<PyVocab>, unk_token: Option<String>) -> PyResult<(Self, PyModel)> {
        let mut builder = WordLevel::builder();

        if let Some(vocab) = vocab {
            match vocab {
                PyVocab::Vocab(vocab) => {
                    builder = builder.vocab(vocab);
                }
                PyVocab::Filename(vocab_filename) => {
                    deprecation_warning(
                        "0.9.0",
                        "WordLevel.__init__ will not create from files anymore, \
                            try `WordLevel.from_file` instead",
                    )?;
                    builder = builder.files(vocab_filename.to_string());
                }
            };
        }
        if let Some(unk_token) = unk_token {
            builder = builder.unk_token(unk_token);
        }

        Ok((
            PyWordLevel {},
            builder
                .build()
                .map_err(|e| exceptions::PyException::new_err(e.to_string()))?
                .into(),
        ))
    }

    #[staticmethod]
    fn read_file(vocab_filename: &str) -> PyResult<Vocab> {
        WordLevel::read_file(vocab_filename).map_err(|e| {
            exceptions::PyValueError::new_err(format!("Error while reading WordLevel file: {}", e))
        })
    }

    #[staticmethod]
    #[args(unk_token = "None")]
    fn from_file(
        py: Python,
        vocab_filename: &str,
        unk_token: Option<String>,
    ) -> PyResult<Py<Self>> {
        let vocab = WordLevel::read_file(vocab_filename).map_err(|e| {
            exceptions::PyValueError::new_err(format!("Error while reading WordLevel file: {}", e))
        })?;
        Py::new(
            py,
            PyWordLevel::new(Some(PyVocab::Vocab(vocab)), unk_token)?,
        )
    }
}

#[pyclass(extends=PyModel, module = "tokenizers.models", name=Unigram)]
pub struct PyUnigram {}

#[pymethods]
impl PyUnigram {
    #[new]
    fn new(vocab: Option<Vec<(String, f64)>>, unk_id: Option<usize>) -> PyResult<(Self, PyModel)> {
        match (vocab, unk_id) {
            (Some(vocab), unk_id) => {
                let model = Unigram::from(vocab, unk_id).map_err(|e| {
                    exceptions::PyException::new_err(format!("Error while loading Unigram: {}", e))
                })?;
                Ok((PyUnigram {}, model.into()))
            }
            (None, None) => Ok((PyUnigram {}, Unigram::default().into())),
            _ => Err(exceptions::PyValueError::new_err(
                "`vocab` and `unk_id` must be both specified",
            )),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::models::PyModel;
    use pyo3::prelude::*;
    use tk::models::bpe::BPE;
    use tk::models::ModelWrapper;

    #[test]
    fn get_subtype() {
        let py_model = PyModel::from(BPE::default());
        let py_bpe = py_model.get_as_subtype().unwrap();
        let gil = Python::acquire_gil();
        assert_eq!(
            "tokenizers.models.BPE",
            py_bpe.as_ref(gil.python()).get_type().name()
        );
    }

    #[test]
    fn serialize() {
        let rs_bpe = BPE::default();
        let rs_bpe_ser = serde_json::to_string(&rs_bpe).unwrap();
        let rs_wrapper: ModelWrapper = rs_bpe.into();
        let rs_wrapper_ser = serde_json::to_string(&rs_wrapper).unwrap();

        let py_model = PyModel::from(rs_wrapper);
        let py_ser = serde_json::to_string(&py_model).unwrap();
        assert_eq!(py_ser, rs_bpe_ser);
        assert_eq!(py_ser, rs_wrapper_ser);

        let py_model: PyModel = serde_json::from_str(&rs_bpe_ser).unwrap();
        match *py_model.model.as_ref().read().unwrap() {
            ModelWrapper::BPE(_) => (),
            _ => panic!("Expected Bert postprocessor."),
        };

        let py_model: PyModel = serde_json::from_str(&rs_wrapper_ser).unwrap();
        match *py_model.model.as_ref().read().unwrap() {
            ModelWrapper::BPE(_) => (),
            _ => panic!("Expected Bert postprocessor."),
        };
    }
}

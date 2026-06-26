# TEST_GUIDE.md

## How to run

```bash
# All tests
pytest -v

# Unit tests only — no dvc repro needed
pytest tests/test_engineer.py tests/test_preprocess.py tests/test_monitor.py -v

# Integration tests — rebuilds pipeline internally from raw CSV
pytest tests/test_pipeline.py tests/test_model.py -v

# With timeout
pytest -v --timeout=60
```

---

## `tests/test_engineer.py`

**Source file tested:** `src/features/engineer.py` — `add_features()`  
**Requires dvc repro:** No  
**Approach:** Calls `add_features()` directly on synthetic DataFrames and asserts on the returned result

### Fixtures (sample data)

**`sample_df`** — 3-row DataFrame representing data after binary/gender encoding but before feature engineering. All 16 original columns present with numeric values (Gender=1/0, CAEC=0/1/2, MTRANS as string).

```
Gender  Age   Height  Weight  family_history  ...  MTRANS
1       25.0  1.75    70.0    1               ...  Public_Transportation
0       32.0  1.60    55.0    0               ...  Walking
1       45.0  1.80    90.0    1               ...  Automobile
```

**`single_row_df`** — 1-row DataFrame for deterministic value checks. Used when a specific calculated value needs to be verified exactly (e.g. BMI = 80 / 1.80²).

---

### `TestOutputColumns` — tests that the correct columns are produced

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_all_new_columns_present` | All 5 engineered columns exist in output | Iterates expected column list, asserts each in `result.columns` | `sample_df` (3 rows) |
| `test_original_columns_preserved` | No original columns dropped | Iterates input columns, asserts each still in output | `sample_df` (3 rows) |
| `test_column_count_increases_by_five` | Exactly 5 new columns added | `result.shape[1] == input.shape[1] + 5` | `sample_df` (3 rows) |
| `test_row_count_unchanged` | Row count unchanged after engineering | `result.shape[0] == input.shape[0]` | `sample_df` (3 rows) |

---

### `TestBMI` — tests BMI = Weight / Height²

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_bmi_formula` | Exact BMI formula correctness | Computes expected = 80.0 / 1.80², asserts `result["BMI"].iloc[0] ≈ expected` with `rel=1e-6` | `single_row_df` (Weight=80, Height=1.80) |
| `test_bmi_all_positive` | BMI never negative or zero | `(result["BMI"] > 0).all()` | `sample_df` (3 rows) |
| `test_bmi_reasonable_range` | BMI stays between 10 and 60 for realistic inputs | `result["BMI"].between(10, 60).all()` | `sample_df` (3 rows) |
| `test_bmi_vectorised_correctly` | All rows computed correctly, not just first | `pd.testing.assert_series_equal(result["BMI"], Weight / Height²)` across all rows | `sample_df` (3 rows) |

---

### `TestHeightGenderInteraction` — tests height_gender_interaction = Height × Gender

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_formula` | Exact formula: Height × Gender | Computes 1.80 × 1, asserts `result.iloc[0] ≈ 1.80` | `single_row_df` (Height=1.80, Gender=1) |
| `test_zero_for_female` | Female (Gender=0) gives 0 | Filters rows where Gender=0, asserts all interaction values == 0 | `sample_df` row 2 (Gender=0) |
| `test_equals_height_for_male` | Male (Gender=1) gives value equal to Height | `pd.testing.assert_series_equal` of interaction vs Height column for male rows | `sample_df` rows 1 and 3 (Gender=1) |

---

### `TestAgeWeightInteraction` — tests age_weight_interaction = Age × Weight

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_formula` | Exact formula: Age × Weight | Computes 30.0 × 80.0 = 2400, asserts result matches | `single_row_df` (Age=30, Weight=80) |
| `test_all_positive` | All interaction values positive | `(result["age_weight_interaction"] > 0).all()` | `sample_df` (3 rows) |
| `test_vectorised_correctly` | All rows computed correctly | `pd.testing.assert_series_equal(result, Age * Weight)` | `sample_df` (3 rows) |

---

### `TestWeightHeightRatio` — tests weight_height_ratio = Weight / Height

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_formula` | Exact formula: Weight / Height | Computes 80.0 / 1.80, asserts match with `rel=1e-6` | `single_row_df` (Weight=80, Height=1.80) |
| `test_all_positive` | All ratios positive | `(result["weight_height_ratio"] > 0).all()` | `sample_df` (3 rows) |
| `test_vectorised_correctly` | All rows computed correctly | `pd.testing.assert_series_equal(result, Weight / Height)` | `sample_df` (3 rows) |

---

### `TestFamilyHistoryWeightRisk` — tests family_history_weight_risk = family_history × Weight

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_formula` | Exact formula: family_history × Weight | Computes 1 × 80.0 = 80.0, asserts match | `single_row_df` (family_history=1, Weight=80) |
| `test_zero_when_no_family_history` | family_history=0 gives risk of 0 | Filters rows where family_history=0, asserts risk == 0 | `sample_df` row 2 (family_history=0) |
| `test_equals_weight_when_family_history` | family_history=1 gives risk equal to Weight | `pd.testing.assert_series_equal` of risk vs Weight for history=1 rows | `sample_df` rows 1 and 3 |

---

### `TestImmutability` — tests input DataFrame is not modified

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_original_df_not_modified` | Input columns and shape unchanged after call | Saves original column list and shape before call, compares after | `sample_df` |
| `test_returns_new_dataframe` | Function returns a new object, not same reference | `assert result is not sample_df` | `sample_df` |

---

### `TestNoNulls` — tests no NaN values introduced

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_no_nulls_in_new_columns` | The 5 engineered columns have no NaN | Selects new columns, `isnull().sum().sum() == 0` | `sample_df` |
| `test_no_new_nulls_introduced` | Entire output has no NaN | `result.isnull().sum().sum() == 0` | `sample_df` |

---

### `TestDtypes` — tests output column data types

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_bmi_is_float` | BMI dtype is `np.float64` | `result["BMI"].dtype == np.float64` | `sample_df` |
| `test_all_new_cols_are_numeric` | All 5 engineered columns are numeric | Iterates new columns, `pd.api.types.is_numeric_dtype()` | `sample_df` |

---
---

## `tests/test_preprocess.py`

**Source file tested:** `src/features/preprocess.py` — `ObesityPreprocessor` class  
**Requires dvc repro:** No  
**Approach:** Instantiates a fresh `ObesityPreprocessor`, fits it on a synthetic 5-row raw DataFrame, then transforms and asserts on results

### Fixtures (sample data)

**`raw_df`** — 5-row DataFrame with raw string categorical values as they appear in the original dataset, before any encoding.

```
Gender   Age   Height  Weight  family_history  FAVC  CAEC         MTRANS
Male     25.0  1.75    70.0    yes             yes   Sometimes    Public_Transportation
Female   32.0  1.60    55.0    no              no    no           Walking
Male     45.0  1.80    90.0    yes             yes   Frequently   Automobile
Female   28.0  1.65    62.0    yes             no    Always       Public_Transportation
Male     36.0  1.70    85.0    no              yes   Sometimes    Walking
```

**`fitted_preprocessor`** — `ObesityPreprocessor` already fitted on `raw_df`. Used by all transform tests.

**`single_row`** — 1-row raw DataFrame for single-row transform shape checks.

---

### `TestSklearnInterface` — tests sklearn API compliance

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_is_base_estimator` | Class inherits `BaseEstimator` | `isinstance(ObesityPreprocessor(), BaseEstimator)` | Fresh instance |
| `test_is_transformer_mixin` | Class inherits `TransformerMixin` | `isinstance(ObesityPreprocessor(), TransformerMixin)` | Fresh instance |
| `test_fit_returns_self` | `fit()` returns the same instance | `result = p.fit(raw_df); assert result is p` | `raw_df` (5 rows) |
| `test_fit_transform_returns_dataframe` | `fit_transform()` returns a DataFrame | `isinstance(result, pd.DataFrame)` | `raw_df` (5 rows) |
| `test_transform_returns_dataframe` | `transform()` returns a DataFrame | `isinstance(result, pd.DataFrame)` | Fitted preprocessor + `raw_df` |
| `test_unfitted_has_none_attributes` | Fresh instance has None state | Checks `scale_cols`, `mtrans_columns`, `feature_columns` all None | Fresh instance |

---

### `TestFitState` — tests that `fit()` sets internal state correctly

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_scale_cols_set_after_fit` | `scale_cols` is a non-empty list after fit | `fitted.scale_cols is not None and len > 0` | Fitted preprocessor |
| `test_mtrans_columns_set_after_fit` | `mtrans_columns` is a non-empty list | `fitted.mtrans_columns is not None and len > 0` | Fitted preprocessor |
| `test_feature_columns_set_after_fit` | `feature_columns` is a non-empty list | `fitted.feature_columns is not None and len > 0` | Fitted preprocessor |
| `test_mtrans_not_in_scale_cols` | MTRANS excluded from StandardScaler | `"MTRANS" not in fitted.scale_cols` | Fitted preprocessor |
| `test_scaler_fitted` | StandardScaler has been fitted (has `mean_`) | `hasattr(fitted.scaler, "mean_")` | Fitted preprocessor |

---

### `TestBinaryEncoding` — tests yes/no columns encoded to numeric

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_binary_cols_are_numeric` | `family_history`, `FAVC`, `SMOKE`, `SCC` are numeric after transform | `pd.api.types.is_numeric_dtype()` for each column | `raw_df` (5 rows) |
| `test_binary_values_only_zero_or_one` | Binary columns contain only 0 or 1 (before scaling) | Checks dtype is numeric (post-scaling values shift slightly) | `raw_df` (5 rows) |

---

### `TestGenderEncoding` — tests Gender encoded to numeric

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_gender_is_numeric_after_transform` | Gender column is numeric (Male=1, Female=0) | `pd.api.types.is_numeric_dtype(result["Gender"])` | `raw_df` (5 rows) |
| `test_gender_col_present` | Gender column still in output | `"Gender" in result.columns` | `raw_df` (5 rows) |

---

### `TestOrdinalEncoding` — tests CAEC and CALC ordinal encoding

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_caec_is_numeric` | CAEC encoded to number (no=0, Sometimes=1, Frequently=2, Always=3) | `pd.api.types.is_numeric_dtype(result["CAEC"])` | `raw_df` (5 rows) |
| `test_calc_is_numeric` | CALC encoded to number | `pd.api.types.is_numeric_dtype(result["CALC"])` | `raw_df` (5 rows) |
| `test_caec_col_present` | CAEC still in output | `"CAEC" in result.columns` | `raw_df` (5 rows) |
| `test_calc_col_present` | CALC still in output | `"CALC" in result.columns` | `raw_df` (5 rows) |

---

### `TestMTRANSEncoding` — tests MTRANS one-hot encoding

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_mtrans_raw_col_dropped` | Original MTRANS string column removed | `"MTRANS" not in result.columns` | `raw_df` (5 rows) |
| `test_mtrans_dummies_present` | At least one MTRANS_ column in output | `any(c.startswith("MTRANS_") for c in result.columns)` | `raw_df` (5 rows) |
| `test_mtrans_dummies_are_numeric` | All MTRANS_ dummy columns are numeric | Iterates MTRANS_ cols, checks dtype | `raw_df` (5 rows) |
| `test_unseen_mtrans_category_handled` | Unseen MTRANS value does not crash transform | Passes row with `MTRANS="Bike"` (not in training), asserts output shape is 1 | 1 row with MTRANS=Bike |

---

### `TestEngineeredFeatures` — tests engineered columns present after transform

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_engineered_cols_present` | All 5 engineered columns in output | Iterates expected list, asserts each in `result.columns` | `raw_df` (5 rows) |
| `test_bmi_present_and_numeric` | BMI column exists and is numeric | `"BMI" in result.columns` and `is_numeric_dtype` | `raw_df` (5 rows) |

---

### `TestOutputShape` — tests output dimensions

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_row_count_preserved` | Number of rows unchanged | `result.shape[0] == raw_df.shape[0]` | `raw_df` (5 rows) |
| `test_output_column_count_greater_than_input` | Output has more columns than input (one-hot + engineering) | `result.shape[1] > raw_df.shape[1]` | `raw_df` (5 rows) |
| `test_feature_columns_consistent` | Two transforms of same data produce identical columns | Transforms twice, `list(r1.columns) == list(r2.columns)` | `raw_df` (5 rows) |
| `test_single_row_transform_shape` | Single row transform produces 1 row × feature_columns cols | `result.shape == (1, len(fitted.feature_columns))` | `single_row` |

---

### `TestNoNulls` — tests no NaN values after transform

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_no_nulls_in_transform_output` | Full transform output has zero NaN | `result.isnull().sum().sum() == 0` | `raw_df` (5 rows) |
| `test_no_nulls_single_row` | Single row transform has zero NaN | `result.isnull().sum().sum() == 0` | `single_row` |

---

### `TestNoDataLeakage` — tests scaler fitted on train only

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_transform_uses_train_scaler_only` | Scaler fitted on first 3 rows can transform last 2 rows without error | Splits `raw_df` into train (rows 0-2) and test (rows 3-4), fits on train, transforms test | `raw_df` split 3/2 |
| `test_reindex_fills_missing_mtrans_with_zero` | Test rows with unseen MTRANS get 0 in those dummy columns | Fits on full `raw_df` (multiple MTRANS), transforms test where all rows have same MTRANS, checks non-matching MTRANS_ cols are all 0 | `raw_df` modified |

---

### `TestColumnConsistency` — tests column order determinism

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_transform_output_matches_feature_columns` | Output columns exactly match `fitted.feature_columns` | `list(result.columns) == fitted.feature_columns` | `raw_df` (5 rows) |
| `test_feature_columns_length_matches_output` | Length of `feature_columns` matches output width | `result.shape[1] == len(fitted.feature_columns)` | `raw_df` (5 rows) |

---
---

## `tests/test_pipeline.py`

**Source files tested:** `src/features/preprocess.py`, `src/features/engineer.py`, `src/data/ingest.py`, `src/models/train.py`  
**Requires dvc repro:** No — rebuilds the full pipeline from the raw CSV inside the fixture  
**Approach:** The `pipeline_artifacts` fixture loads the real dataset, splits it, fits the preprocessor, transforms data, and trains a RandomForest. All test classes share this fixture via `scope="module"` so it runs once per test session.

### Fixture (sample data)

**`raw_data`** — the full cleaned dataset: 2111 rows − 24 duplicates = 2087 rows, 17 columns.

**`pipeline_artifacts`** — module-scoped dict containing:

```
preprocessor  → ObesityPreprocessor fitted on X_train (1669 rows)
model         → RandomForestClassifier trained on train_df
X_train       → 1669 rows × 16 features (raw, before transform)
X_test        → 418 rows × 16 features (raw, before transform)
y_train       → 1669 labels
y_test        → 418 labels
train_df      → 1669 rows × 26 features + NObeyesdad (transformed)
test_df       → 418 rows × 26 features + NObeyesdad (transformed)
```

---

### `TestRawData` — tests the raw dataset loaded from disk

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_raw_file_exists` | CSV file exists at configured path | `DATA_RAW.exists()` | File on disk |
| `test_raw_data_loads` | `clean(load_raw())` returns a DataFrame | `isinstance(raw_data, pd.DataFrame)` | 2087 rows |
| `test_raw_data_has_rows` | Dataset is non-empty | `len(raw_data) > 0` | 2087 rows |
| `test_raw_data_has_expected_columns` | All 17 expected columns present | Iterates expected list, asserts each in `raw_data.columns` | 2087 rows |
| `test_target_column_present` | `NObeyesdad` exists | `"NObeyesdad" in raw_data.columns` | 2087 rows |
| `test_target_has_seven_classes` | Exactly 7 obesity classes | `raw_data["NObeyesdad"].nunique() == 7` | 2087 rows |
| `test_no_duplicates_after_clean` | `clean()` removed all duplicates | `raw_data.duplicated().sum() == 0` | 2087 rows |
| `test_no_nulls_in_raw` | No missing values | `raw_data.isnull().sum().sum() == 0` | 2087 rows |

---

### `TestTrainTestSplit` — tests 80/20 stratified split correctness

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_split_sizes` | Test set size ≈ 20% of total (±2 rows) | `abs(len(X_test) - round(n_total × 0.2)) <= 2` | 2087 rows split to 1669/418 |
| `test_no_overlap_between_train_and_test` | No row index shared between train and test | `train_idx.isdisjoint(test_idx)` | Index sets of X_train and X_test |
| `test_train_test_together_equal_full` | Train + test = full dataset size | `n_train + n_test == len(raw_data)` | 1669 + 418 = 2087 |
| `test_stratified_class_distribution` | Each class proportion in train within 5% of full dataset | Computes `value_counts(normalize=True)` for full and train, checks each class diff < 0.05 | All 7 classes |
| `test_random_state_reproducible` | Same `random_state=42` produces identical splits across two calls | Splits twice with same params, `pd.testing.assert_frame_equal` | 2087 rows |

---

### `TestPreprocessorOutput` — tests preprocessor transforms train and test correctly

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_train_df_has_target` | `NObeyesdad` column in train_df | `"NObeyesdad" in train_df.columns` | 1669 rows |
| `test_test_df_has_target` | `NObeyesdad` column in test_df | `"NObeyesdad" in test_df.columns` | 418 rows |
| `test_train_df_no_nulls` | No NaN in train_df | `train_df.isnull().sum().sum() == 0` | 1669 rows |
| `test_test_df_no_nulls` | No NaN in test_df | `test_df.isnull().sum().sum() == 0` | 418 rows |
| `test_train_test_same_feature_columns` | Feature columns identical between train and test | Compares column lists excluding NObeyesdad | 1669 vs 418 rows |
| `test_mtrans_raw_col_not_in_output` | Raw MTRANS string column replaced by dummies | `"MTRANS" not in train_df.columns` | Both DataFrames |
| `test_bmi_col_present` | BMI engineered column present | `"BMI" in train_df.columns` | 1669 rows |
| `test_engineered_cols_in_train` | All 5 engineered columns present | Iterates expected list, asserts each in `train_df.columns` | 1669 rows |

---

### `TestNoDataLeakage` — tests scaler fitted only on training data

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_scaler_fitted_on_train_only` | StandardScaler has `mean_` attribute (is fitted) | `hasattr(pre.scaler, "mean_") and pre.scaler.mean_ is not None` | Fitted preprocessor |
| `test_fit_transform_vs_transform_differ_on_test` | Scaling X_test with train-fitted scaler differs from scaling X_test with test-fitted scaler | Transforms X_test with train preprocessor vs fresh preprocessor fitted on X_test, asserts results are NOT equal | X_test (418 rows) |
| `test_test_indices_not_in_train` | No row index overlap | `len(train_idx & test_idx) == 0` | Index sets |

---

### `TestArtifacts` — tests objects are correct types and serialisable

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_preprocessor_is_obesity_preprocessor` | Preprocessor is correct class | `isinstance(preprocessor, ObesityPreprocessor)` | Fixture object |
| `test_model_is_random_forest` | Model is RandomForestClassifier | `isinstance(model, RandomForestClassifier)` | Fixture object |
| `test_model_has_classes` | Model knows 7 classes | `len(model.classes_) == 7` | Fixture object |
| `test_model_feature_count_matches_preprocessor` | Model input count matches preprocessor output count | `model.n_features_in_ == len(preprocessor.feature_columns)` | Both objects |
| `test_pipeline_pkl_serialisable` | Preprocessor survives pickle round-trip | Pickles to `tmp_path`, unpickles, checks type and `feature_columns` match | Temp file |
| `test_model_pkl_serialisable` | Model survives pickle round-trip | Pickles to `tmp_path`, unpickles, checks type | Temp file |
| `test_feature_columns_json_serialisable` | Feature column list survives JSON round-trip | Writes and reads JSON, asserts lists equal | Temp file |

---

### `TestModelTraining` — tests model prediction behaviour

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_model_predicts_on_test` | `model.predict()` returns one prediction per row | `len(preds) == len(X_test)` | 418 rows |
| `test_predictions_are_valid_classes` | All predictions are one of the 7 known classes | `all(p in model.classes_ for p in preds)` | 418 rows |
| `test_predict_proba_shape` | `predict_proba()` returns (418, 7) array | `proba.shape == (418, 7)` | 418 rows |
| `test_predict_proba_sums_to_one` | Each row's probabilities sum to 1.0 | `np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)` | 418 rows |
| `test_model_accuracy_above_threshold` | Accuracy ≥ 0.95 on test set | Computes `accuracy_score(y_test, preds)`, asserts ≥ threshold from `params.yaml` | 418 rows |

---

### `TestReproducibility` — tests identical results across runs

| Test | What is tested | How | Sample |
|---|---|---|---|
| `test_same_split_same_random_state` | Same `random_state` produces identical splits | Splits twice with `random_state=42`, `pd.testing.assert_frame_equal` on both train and test | 2087 rows |
| `test_same_model_params_same_predictions` | Training twice with same params → identical predictions | Trains two RandomForests with `n_jobs=1` and same seed, `np.testing.assert_array_equal` | 418 rows |

---
---

## `tests/test_model.py`

**Source files tested:** Same as `test_pipeline.py` — full pipeline rebuild  
**Requires dvc repro:** No — rebuilds pipeline from raw CSV  
**Approach:** Identical `pipeline_artifacts` fixture as `test_pipeline.py`. Same test classes with identical logic. Both files share the same 7 test classes covering `TestRawData`, `TestTrainTestSplit`, `TestPreprocessorOutput`, `TestNoDataLeakage`, `TestArtifacts`, `TestModelTraining`, and `TestReproducibility`.

All tests, samples, and implementation details are identical to `test_pipeline.py` above.

> Note: `test_model.py` and `test_pipeline.py` currently contain duplicate test logic. They exist as separate files for clarity of ownership — `test_pipeline.py` is owned by the QA lead covering data and preprocessing integrity, while `test_model.py` covers model training and prediction behaviour.

---
---

## `tests/test_monitor.py`

**Source file tested:** `src/monitoring/monitor.py` — `check_data_drift()`  
**Requires dvc repro:** No — creates its own CSV files at runtime  
**Approach:** Each test writes controlled CSV files to `data/reference/` and `monitoring/logs/` then calls `check_data_drift()` and asserts on the returned dict

### Fixtures (sample data)

**`setup_dirs`** — `autouse=True`, runs before every test. Creates `DATA_REF.parent`, `PROD_LOG.parent`, and `DRIFT_REPORT.parent` directories if they do not exist.

**`reference_data`** — writes 200 rows to `data/reference/training_reference.csv` with BMI values uniformly spread between 21 and 29, representing normal weight training distribution.

```
BMI    NObeyesdad
22.0   Normal_Weight
25.0   Normal_Weight
28.0   Normal_Weight
...    (200 rows, BMI range 21–29)
```

**`similar_prod`** — depends on `reference_data`. Writes 200 rows to `monitoring/logs/production_prediction_log.csv` with BMI values between 21.5 and 29.5, overlapping heavily with reference.

```
BMI    prediction
22.5   Normal_Weight
25.5   Normal_Weight
...    (200 rows, BMI range 21.5–29.5)
```

**`drifted_prod`** — depends on `reference_data`. Writes 200 rows with BMI values between 38 and 46, completely separated from reference distribution.

```
BMI    prediction
40.0   Obesity_Type_III
42.0   Obesity_Type_III
...    (200 rows, BMI range 38–46)
```

---

### Tests

| Test | What is tested | How | Input fixtures |
|---|---|---|---|
| `test_no_drift_detected` | KS-test correctly identifies no drift when distributions overlap | Calls `check_data_drift()`, asserts `result["drift_detected"] is False` | `reference_data` + `similar_prod` (BMI 21–29 vs 21.5–29.5) |
| `test_drift_detected` | KS-test correctly identifies drift when distributions are separated | Calls `check_data_drift()`, asserts `result["drift_detected"] is True` | `reference_data` + `drifted_prod` (BMI 21–29 vs 38–46) |
| `test_result_has_p_value` | Return dict always contains `p_value` key with valid probability | Calls `check_data_drift()`, asserts `"p_value" in result` and `0.0 <= result["p_value"] <= 1.0` | `reference_data` + `similar_prod` |
| `test_result_has_ks_stat` | Return dict always contains `ks_stat` key with non-negative value | Calls `check_data_drift()`, asserts `"ks_stat" in result` and `result["ks_stat"] >= 0.0` | `reference_data` + `similar_prod` |
| `test_no_prod_log_returns_no_drift` | Function returns gracefully when production log does not exist | Deletes `PROD_LOG` if it exists, calls `check_data_drift()`, asserts `drift_detected=False` and `"reason" in result` | `reference_data` only, no prod log |
| `test_drift_report_written` | `drift_report.json` is written to disk after every check | Deletes existing report, calls `check_data_drift()`, asserts file exists and contains `"drift_detected"` key | `reference_data` + `similar_prod` |
| `test_p_value_range` | p-value is always a valid probability (0 to 1) | Calls `check_data_drift()`, asserts `0.0 <= result["p_value"] <= 1.0` | `reference_data` + `similar_prod` |

---

## Dependency summary

```
test_engineer   → src/features/engineer.py
                  Synthetic 3-row and 1-row DataFrames (no files)

test_preprocess → src/features/preprocess.py (ObesityPreprocessor)
                  src/features/engineer.py
                  Synthetic 5-row raw DataFrame (no files)

test_pipeline   → src/features/preprocess.py
                  src/features/engineer.py
                  src/data/ingest.py
                  data/raw/ObesityDataSet_raw_and_data_sinthetic.csv (real file)
                  params.yaml

test_model      → src/features/preprocess.py
                  src/features/engineer.py
                  src/data/ingest.py
                  data/raw/ObesityDataSet_raw_and_data_sinthetic.csv (real file)
                  params.yaml

test_monitor    → src/monitoring/monitor.py
                  data/reference/training_reference.csv (created by fixture)
                  monitoring/logs/production_prediction_log.csv (created by fixture)
                  params.yaml
```

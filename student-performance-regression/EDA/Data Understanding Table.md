## Dataset Variables

| **Variable Name** | **Data Type** | **Range / Categories** | **Description** |
|---|---|---|---|
| `age` | Numeric | 15 – 22 | Student's age |
| `studytime` | Numeric | 1 (<2 hrs), 2 (2–5 hrs), 3 (5–10 hrs), 4 (>10 hrs) | Weekly study time |
| `failures` | Numeric | 0, 1, 2, 3 (if ≥ 3) | Number of past class failures |
| `absences` | Numeric | 0 – 93 | Number of school absences |
| `G1` | Numeric | 0 – 20 | First period grade |
| `G2` | Numeric | 0 – 20 | Second period grade |
| `G3` | Numeric | 0 – 20 | Final grade (**Target Variable**) |








## Data Quality and Statistical Insights

* **Data Quality:** The dataset contains 395 rows and 33 columns with no missing values.
* **Grade Distribution:** `G3` shows the distribution of students' final grades.
* **Correlation:** `G1` and `G2` have the strongest positive relationship with `G3`, while `failures` show a negative relationship.
* **Preprocessing:** Categorical features require One-Hot Encoding, and `G1` and `G2` must be considered carefully to avoid data leakage.

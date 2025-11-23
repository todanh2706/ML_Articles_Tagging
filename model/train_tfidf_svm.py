import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report

def main():
    # 1) Load data
    def load_split(x_path, y_path):
        X = pd.read_csv(x_path, encoding="utf-8-sig")
        y = pd.read_csv(y_path, encoding="utf-8-sig")
        return X, y

    X_train, y_train = load_split("./train/X_train_basic.csv", "./train/y_train_basic.csv")
    X_val,   y_val   = load_split("./val/X_val_basic.csv",   "./val/y_val_basic.csv")
    X_test,  y_test  = load_split("./test/X_test_basic.csv",  "./test/y_test_basic.csv")

    # 2) Chọn cột text
    TEXT_COL = "content_final"

    train_texts = X_train[TEXT_COL].astype(str)
    val_texts   = X_val[TEXT_COL].astype(str)
    test_texts  = X_test[TEXT_COL].astype(str)

    # 3) Lấy nhãn label_encoded
    LABEL_COL = "label_encoded"
    train_labels = y_train[LABEL_COL]
    val_labels   = y_val[LABEL_COL]
    test_labels  = y_test[LABEL_COL]

    # Bỏ NaN trong nhãn
    train_mask = train_labels.notna()
    val_mask   = val_labels.notna()
    test_mask  = test_labels.notna()

    train_texts, train_labels = train_texts[train_mask], train_labels[train_mask]
    val_texts,   val_labels   = val_texts[val_mask],     val_labels[val_mask]
    test_texts,  test_labels  = test_texts[test_mask],   test_labels[test_mask]

    # 4) Pipeline TF-IDF + LinearSVC
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            max_df=0.9,
            min_df=2,
            sublinear_tf=True
        )),
        ("svm", LinearSVC(
            class_weight="balanced"
        ))
    ])

    # 5) Tune hyperparameters trên TRAIN (cv nội bộ), chọn theo f1_macro
    param_grid = {
        "tfidf__ngram_range": [(1,1), (1,2)],
        "tfidf__max_df": [0.8, 0.9, 1.0],
        "svm__C": [0.1, 0.5, 1.0, 2.0, 5.0]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(train_texts, train_labels)

    print("Best params:", grid.best_params_)
    print("Best CV f1_macro:", grid.best_score_)

    # 6) Đánh giá trên VAL
    val_pred = grid.predict(val_texts)
    print("\n=== VALIDATION ===")
    print("VAL accuracy:", accuracy_score(val_labels, val_pred))
    print("VAL f1_macro:", f1_score(val_labels, val_pred, average="macro"))
    print(classification_report(val_labels, val_pred))

    # 7) Retrain trên TRAIN+VAL rồi test
    best_model = grid.best_estimator_

    all_texts  = pd.concat([train_texts, val_texts], axis=0)
    all_labels = pd.concat([train_labels, val_labels], axis=0)

    best_model.fit(all_texts, all_labels)

    test_pred = best_model.predict(test_texts)

    print("\n=== TEST ===")
    print("TEST accuracy:", accuracy_score(test_labels, test_pred))
    print("TEST f1_macro:", f1_score(test_labels, test_pred, average="macro"))
    print(classification_report(test_labels, test_pred))

if __name__ == "__main__":
    main()
import os
import argparse
import numpy as np

from src.data_loader         import load_csv, split_data
from src.preprocessor        import TextPreprocessor
from src.feature_engineering import FeatureExtractor
from src.models              import ModelTrainer, MODEL_REGISTRY
from src.evaluator           import Evaluator
from src.predictor           import SentimentPredictor


def parse_args():
    p = argparse.ArgumentParser(description="Sentiment Analysis — Training Pipeline")
    p.add_argument("--csv",       type=str, default=None)
    p.add_argument("--text_col",  type=str, default="text")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--model",     type=str, default="logistic_regression",
                   choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--strategy",  type=str, default="tfidf_word",
                   choices=["tfidf_word", "tfidf_char", "tfidf_combo", "bow"])
    p.add_argument("--tune",      action="store_true")
    p.add_argument("--compare",   action="store_true")
    p.add_argument("--output",    type=str, default="outputs")
    return p.parse_args()


def _get_coefficients(trainer):
    model = trainer.get_model()
    if hasattr(model, "calibrated_classifiers_"):
        inner = model.calibrated_classifiers_[0].estimator
        coef  = getattr(inner, "coef_", None)
        return coef.ravel() if coef is not None else np.zeros(1)
    coef = getattr(model, "coef_", None)
    return coef.ravel() if coef is not None else None


def run_pipeline(args):
    os.makedirs(args.output, exist_ok=True)

    if args.csv:
        df = load_csv(args.csv, text_col=args.text_col, label_col=args.label_col)
    else:
        raise ValueError("Please provide a CSV file using --csv.")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print("\n[Pipeline] Preprocessing text...")
    preprocessor  = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    X_train_clean = preprocessor.transform(X_train)
    X_val_clean   = preprocessor.transform(X_val)
    X_test_clean  = preprocessor.transform(X_test)

    print("\n[Pipeline] Extracting features...")
    fe          = FeatureExtractor(strategy=args.strategy, max_features=50_000)
    X_train_vec = fe.fit_transform(X_train_clean)
    X_val_vec   = fe.transform(X_val_clean)
    X_test_vec  = fe.transform(X_test_clean)

    evaluator = Evaluator(output_dir=args.output)

    if args.compare:
        print("\n[Pipeline] Benchmarking all classifiers...")
        all_results = {}
        for name in MODEL_REGISTRY:
            print(f"\n{'─'*45}")
            t     = ModelTrainer(model_name=name)
            t.train(X_train_vec, y_train)
            preds = t.predict(X_val_vec)
            try:
                probs = t.predict_proba(X_val_vec)
            except AttributeError:
                probs = None
            metrics            = evaluator.evaluate(y_val, preds, probs, model_name=name)
            all_results[name]  = metrics

        evaluator.compare_models(all_results)
        print("\n[Pipeline] Model comparison complete.")

    print(f"\n[Pipeline] Training primary model: '{args.model}'")
    trainer = ModelTrainer(model_name=args.model)

    if args.tune:
        trainer.hyperparameter_tune(X_train_vec, y_train, cv=5)
    else:
        trainer.train(X_train_vec, y_train)
        trainer.cross_validate(X_train_vec, y_train, cv=5)

    print("\n[Pipeline] Evaluating on VALIDATION set...")
    val_preds = trainer.predict(X_val_vec)
    try:
        val_probs = trainer.predict_proba(X_val_vec)
    except AttributeError:
        val_probs = None

    evaluator.evaluate(y_val, val_preds, val_probs, model_name=f"{args.model}_val")

    print("\n[Pipeline] Evaluating on HELD-OUT TEST set...")
    test_preds = trainer.predict(X_test_vec)
    try:
        test_probs = trainer.predict_proba(X_test_vec)
    except AttributeError:
        test_probs = None

    final_metrics = evaluator.evaluate(
        y_test, test_preds, test_probs, model_name=f"{args.model}_test"
    )

    evaluator.plot_confusion_matrix(y_test, test_preds, model_name=args.model)
    if test_probs is not None:
        evaluator.plot_roc_curve(y_test, test_probs, model_name=args.model)

    coef = _get_coefficients(trainer)
    if coef is not None:
        try:
            feature_names = fe.get_feature_names()
            evaluator.plot_top_features(feature_names, coef, model_name=args.model, top_n=20)
        except Exception as e:
            print(f"[Pipeline] Feature importance plot skipped: {e}")

    save_path = os.path.join(args.output, f"sentiment_pipeline_{args.model}.joblib")
    predictor = SentimentPredictor(preprocessor=preprocessor,
                                   feature_extractor=fe,
                                   trainer=trainer)
    predictor.save(save_path)

    print("\n[Pipeline] Quick inference smoke-test:")
    sample_texts = [
        "This movie was an absolute masterpiece. The acting was superb!",
        "Terrible film. Total waste of time. Boring and predictable.",
        "Not bad, had some good moments but also dragged in parts.",
    ]
    for r in predictor.predict_batch(sample_texts, verbose=False):
        emoji    = "👍" if r["label"] == "POSITIVE" else "👎"
        conf_str = f"{r['confidence']:.1%}" if r["confidence"] is not None else "N/A"
        print(f"  {emoji} [{conf_str}] {r['text'][:65]}...")

    print("\n[Pipeline] All done! Outputs saved to:", args.output)
    return final_metrics


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
import argparse
from src.predictor import SentimentPredictor

def parse_args():
    p = argparse.ArgumentParser(description="Sentiment Analysis — Inference")
    p.add_argument("--model", type=str, required=True,
                   help="Path to saved .joblib pipeline")
    p.add_argument("--text",  type=str, default=None,
                   help="Single review string to classify")
    p.add_argument("--file",  type=str, default=None,
                   help="Path to a .txt file with one review per line")
    return p.parse_args()

def main():
    args = parse_args()
    predictor = SentimentPredictor.load(args.model)

    if args.text:
        predictor.predict_one(args.text, verbose=True)

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        results = predictor.predict_batch(lines, verbose=False)
        print(f"\n{'─'*65}")
        print(f"  {'LABEL':<12} {'CONFIDENCE':<12} TEXT")
        print(f"{'─'*65}")
        for r in results:
            emoji = "👍" if r["label"] == "POSITIVE" else "👎"
            print(f"  {emoji} {r['label']:<10} {r['confidence']:.2%}      "
                  f"{r['text'][:45]}...")
        print(f"{'─'*65}")

    else:
        predictor.interactive_demo()

if __name__ == "__main__":
    main()
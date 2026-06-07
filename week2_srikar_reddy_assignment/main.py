"""Week-2 Tesla deliveries forecasting pipeline."""

from pathlib import Path

from src.features import create_features
from src.models import (
    analyze_errors,
    build_test_predictions,
    compare_models,
    describe_forecast_trend,
    evaluate_model,
    forecast_quarters,
    plot_actual_vs_predicted,
    plot_future_forecast,
    save_model_results,
    train_models,
)
from src.preprocessing import preprocess

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    print("=" * 60)
    print("Tesla Deliveries Forecasting — Week 2")
    print("=" * 60)

    print("\n[1/5] Loading and preprocessing data...")
    quarterly_df = preprocess()
    print(f"  Quarterly records: {len(quarterly_df)}")
    print(f"  Date range: {quarterly_df['date'].min().date()} to {quarterly_df['date'].max().date()}")

    print("\n[2/5] Creating features...")
    featured_df = create_features(quarterly_df)
    print(f"  Rows after feature engineering: {len(featured_df)}")

    print("\n[3/5] Training and evaluating models...")
    results = train_models(featured_df)
    comparison = compare_models(results)
    save_model_results(comparison, PROJECT_ROOT / "model_results.csv")

    print("\nModel Comparison (test set — last 4 quarters):")
    print("-" * 78)
    header = f"{'Model':<22} {'RMSE':>9} {'MAE':>9} {'R2':>7} {'MAPE':>7} {'Accuracy':>9}"
    print(header)
    print("-" * 78)
    for _, row in comparison.iterrows():
        print(
            f"{row['Model']:<22} {row['RMSE']:>9,.0f} {row['MAE']:>9,.0f} "
            f"{row['R2']:>7.3f} {row['MAPE']:>6.2f}% {row['Forecast_Accuracy']:>8.2f}%"
        )

    print("\nHyperparameter tuning results:")
    print(f"  Best Ridge alpha: {results['best_alphas']['Ridge Regression']}")
    print(f"  Best Lasso alpha: {results['best_alphas']['Lasso Regression']}")

    best_name = comparison.iloc[0]["Model"]
    best_model = results["models"][best_name]
    y_pred = best_model.predict(results["x_test"])
    best_metrics = evaluate_model(results["y_test"], y_pred)

    predictions = build_test_predictions(results["test_df"], results["y_test"], y_pred)
    predictions.to_csv(PROJECT_ROOT / "test_predictions.csv", index=False)

    print(f"\nSaved: model_results.csv, test_predictions.csv")

    print("\n[4/5] Error analysis...")
    error_info = analyze_errors(predictions)
    print(error_info["summary"])

    plot_actual_vs_predicted(
        results["test_df"],
        results["y_test"],
        y_pred,
        best_name,
        PROJECT_ROOT / "actual_vs_predicted.png",
    )
    print("Saved: actual_vs_predicted.png")

    print("\n[5/5] Forecasting next 4 quarters...")
    forecast_df = forecast_quarters(quarterly_df, best_model)
    plot_future_forecast(
        quarterly_df,
        forecast_df,
        PROJECT_ROOT / "future_forecast.png",
    )
    print("Saved: future_forecast.png")

    print(f"\nForecast ({best_name}):")
    for _, row in forecast_df.iterrows():
        print(f"  {int(row['year'])}-Q{int(row['quarter'])}: {row['forecast']:,.0f} deliveries")

    trend = describe_forecast_trend(forecast_df)
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{best_name} achieved the best overall performance.")
    print(f"Forecast accuracy was approximately {best_metrics['Forecast_Accuracy']:.0f}%.")
    print(f"Average absolute error on the test set: {error_info['avg_absolute_error']:,.0f} deliveries.")
    print(f"The model predicts {trend}.")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

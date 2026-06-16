import matplotlib.pyplot as plt
from pathlib import Path

from Scripts.utils.calculate_metrics import calculate_metrics


def plot_predictions(
    ticker,
    y_train,
    y_train_pred,
    y_val,
    y_val_pred,
    y_test,
    y_test_pred,
    history,
    plots_dir,
):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Training History (Loss)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title(f'{ticker} - Training History (Loss)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Training History (MAE)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax2.set_title(f'{ticker} - Training History (MAE)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Test Set Predictions vs Actual
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(y_test, label='Actual', linewidth=2, alpha=0.7)
    ax3.plot(y_test_pred, label='Predicted', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Sample', fontsize=12, fontweight='bold')
    ax3.set_ylabel('21-Day Forward Return', fontsize=12, fontweight='bold')
    ax3.set_title(f'{ticker} - Test Set: Predictions vs Actual', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Scatter Plot (Test Set)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(y_test, y_test_pred, alpha=0.5, s=30)

    # Perfect prediction line
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax4.set_xlabel('Actual Returns', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Predicted Returns', fontsize=12, fontweight='bold')
    ax4.set_title(f'{ticker} - Prediction Scatter (Test)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. Residuals Distribution
    ax5 = fig.add_subplot(gs[2, 1])
    residuals = y_test - y_test_pred
    ax5.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax5.set_title(f'{ticker} - Residuals Distribution', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Add overall metrics text
    test_metrics = calculate_metrics(y_test, y_test_pred)
    metrics_text = (
        f"Test Metrics:\n"
        f"R² = {test_metrics['r2']:.4f}\n"
        f"RMSE = {test_metrics['rmse']:.4f}\n"
        f"MAE = {test_metrics['mae']:.4f}\n"
        f"Dir Acc = {test_metrics['direction_accuracy']:.2f}%"
    )

    fig.text(0.98, 0.02, metrics_text, fontsize=11,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save figure
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    filepath = plots_dir / f"{ticker}_predictions.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
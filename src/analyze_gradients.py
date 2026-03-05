import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_gradients(csv_file):

    # -------------------------------
    # Setup output directory
    # -------------------------------
    csv_file = os.path.abspath(csv_file)
    out_dir = os.path.dirname(csv_file)
    base_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Saving outputs to: {out_dir}")

    df = pd.read_csv(csv_file)

    required_cols = [
        "epoch",
        "train_config_gnorm_combined_epoch",
        "train_config_gnorm_diffusion_loss_epoch",
        "train_config_gnorm_voxel_energy_loss_epoch",
        "train_config_gdot_combined__diffusion_loss_epoch",
        "train_config_gdot_combined__voxel_energy_loss_epoch",
        "train_config_gdot_diffusion_loss__voxel_energy_loss_epoch"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    epochs = df["epoch"]

    g_comb = df["train_config_gnorm_combined_epoch"]
    g_diff = df["train_config_gnorm_diffusion_loss_epoch"]
    g_vox  = df["train_config_gnorm_voxel_energy_loss_epoch"]

    dot_cd = df["train_config_gdot_combined__diffusion_loss_epoch"]
    dot_cv = df["train_config_gdot_combined__voxel_energy_loss_epoch"]
    dot_dv = df["train_config_gdot_diffusion_loss__voxel_energy_loss_epoch"]
    print("g_config norm sample:", 
      df["train_config_gnorm_combined_epoch"].describe())
    # ===============================
    # 1. Norm Ratio Analysis
    # ===============================
    ratio_vox_diff = g_vox / (g_diff + 1e-12)

    fig = plt.figure()
    plt.plot(epochs, ratio_vox_diff)
    plt.title("||g_voxel|| / ||g_diffusion||")
    plt.xlabel("Epoch")
    plt.ylabel("Norm Ratio")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{base_name}_ratio_voxel_diffusion.png"))
    plt.close(fig)

    # ===============================
    # 2. Raw Norm Comparison
    # ===============================
    fig = plt.figure()
    plt.plot(epochs, g_comb, label="||g_combined||")
    plt.plot(epochs, g_diff, label="||g_diffusion||")
    plt.plot(epochs, g_vox, label="||g_voxel||")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm Comparison")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{base_name}_norm_comparison.png"))
    plt.close(fig)

    # ===============================
    # 3. Combined Dominance
    # ===============================
    dist_to_diff = np.abs(g_comb - g_diff)
    dist_to_vox  = np.abs(g_comb - g_vox)

    fig = plt.figure()
    plt.plot(epochs, dist_to_diff, label="|g_comb - g_diff|")
    plt.plot(epochs, dist_to_vox, label="|g_comb - g_vox|")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Distance")
    plt.title("Combined Gradient Closeness")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{base_name}_combined_closeness.png"))
    plt.close(fig)

    # ===============================
    # 4. Voxel vs Combined Ratio
    # ===============================
    fig = plt.figure()
    plt.plot(epochs, g_vox / (g_comb + 1e-12))
    plt.title("||g_voxel|| / ||g_combined||")
    plt.xlabel("Epoch")
    plt.ylabel("Ratio")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{base_name}_ratio_voxel_combined.png"))
    plt.close(fig)
    
    # ===============================
    # 5. Cosine Similarity Analysis
    # ===============================
    eps = 1e-8  # larger than 1e-12 — more stable for small norms
    
    cos_sim_diff_vox  = (dot_dv / (g_diff * g_vox  + eps)).clip(-1, 1)
    cos_sim_comb_diff = (dot_cd / (g_comb * g_diff  + eps)).clip(-1, 1)
    cos_sim_comb_vox  = (dot_cv / (g_comb * g_vox   + eps)).clip(-1, 1)
    fig = plt.figure()
    plt.plot(epochs, cos_sim_diff_vox,  label="cos(diffusion, voxel)")
    plt.plot(epochs, cos_sim_comb_diff, label="cos(combined, diffusion)")
    plt.plot(epochs, cos_sim_comb_vox,  label="cos(combined, voxel)")
    plt.axhline(y=0, color='red', linestyle='--', label="conflict boundary")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Similarity")
    plt.title("Gradient Cosine Similarities")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{base_name}_cosine_similarities.png"))
    plt.close(fig)

    
    # ===============================
    # 6. Correlation Matrix
    # ===============================
    corr = pd.DataFrame({
        "g_comb": g_comb,
        "g_diff": g_diff,
        "g_vox": g_vox,
        "dot_cd": dot_cd,
        "dot_cv": dot_cv,
        "dot_dv": dot_dv
    }).corr()

    corr_path = os.path.join(out_dir, f"{base_name}_correlation_matrix.csv")
    corr.to_csv(corr_path)

    # ===============================
    # 6. Summary File
    # ===============================
    summary_text = []
    summary_text.append("Gradient Norm Analysis Summary\n")
    summary_text.append(f"Mean ||g_voxel|| / ||g_diffusion||: {ratio_vox_diff.mean()}\n")
    summary_text.append(f"Max ratio: {ratio_vox_diff.max()}\n")
    summary_text.append(f"Min ratio: {ratio_vox_diff.min()}\n")
    summary_text.append(f"Fraction epochs where combined closer to voxel: {(dist_to_vox < dist_to_diff).mean()}\n")
    conflict_epochs = (cos_sim_diff_vox < 0).sum()
    first_conflict  = epochs[cos_sim_diff_vox < 0].iloc[0] if conflict_epochs > 0 else None

    summary_text.append(f"Mean cos(diffusion, voxel): {cos_sim_diff_vox.mean()}\n")
    summary_text.append(f"Fraction epochs in conflict (cos < 0): {(cos_sim_diff_vox < 0).mean()}\n")
    summary_text.append(f"First conflict epoch: {first_conflict}\n")

    summary_path = os.path.join(out_dir, f"{base_name}_gradient_summary.txt")
    with open(summary_path, "w") as f:
        f.writelines(summary_text)

    print("Analysis complete.")
    print(f"Saved plots and summary with prefix: {base_name}_*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze gradient norm relationships.")
    parser.add_argument("csv_file", type=str, help="Path to training CSV log file")

    args = parser.parse_args()
    analyze_gradients(args.csv_file)
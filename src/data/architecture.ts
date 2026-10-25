export type ArchitectureStage = {
  id: string;
  title: string;
  detail: string;
  execution: "encrypted" | "plaintext" | "training-only";
};

export const baselineArchitecture: ArchitectureStage[] = [
  { id: "input", title: "8×8 image", detail: "64 grayscale pixels", execution: "plaintext" },
  { id: "preprocess", title: "Preprocessing", detail: "Training-fitted StandardScaler or fixed pixel /16", execution: "plaintext" },
  { id: "conv", title: "3×3 valid convolution", detail: "One 6×6 feature map in this reconstruction", execution: "encrypted" },
  { id: "square", title: "Square activation", detail: "Polynomial activation z²", execution: "encrypted" },
  { id: "pool", title: "2×2 average pooling", detail: "Stride 2; 3×3 pooled feature map", execution: "encrypted" },
  { id: "decrypt", title: "Feature decryption", detail: "Pooled feature vector leaves the encrypted domain", execution: "plaintext" },
  { id: "head", title: "Logistic Regression head", detail: "Linear logit and threshold", execution: "plaintext" },
  { id: "prediction", title: "Prediction", detail: "Positive class is digit 1 (or 8 in secondary task)", execution: "plaintext" },
];

export function candidateArchitecture(preprocessing: string, rangeBound: number | null, lambda: number): ArchitectureStage[] {
  const stages: ArchitectureStage[] = [
    { id: "input", title: "8×8 image", detail: "64 grayscale pixels", execution: "plaintext" },
    { id: "preprocess", title: "Input calibration", detail: preprocessing === "pixel_0_1" ? "Fixed raw intensity / 16 → [0,1]" : "StandardScaler fit on training rows only", execution: "plaintext" },
    { id: "conv", title: "3×3 valid convolution", detail: "Single 6×6 map; candidate weights retrained for selected input scale", execution: "encrypted" },
    { id: "square", title: "Square activation", detail: "Encrypted z²; selected test range is reported separately", execution: "encrypted" },
  ];
  if (rangeBound !== null && lambda > 0) stages.push({ id: "range-loss", title: "Range penalty · training only", detail: `λ=${lambda}; target |z| ≤ ${rangeBound}. This loss is not an inference step.`, execution: "training-only" });
  stages.push(
    { id: "pool", title: "2×2 average pooling", detail: "Stride 2; encrypted in mask path, plaintext after early map decryption in im2col path", execution: "encrypted" },
    { id: "decrypt", title: "Feature decryption", detail: "The selected experimental CKKS paths decrypt before the final head", execution: "plaintext" },
    { id: "head", title: "Linear classification head", detail: "Plaintext logit and threshold; not end-to-end encrypted", execution: "plaintext" },
    { id: "prediction", title: "Prediction", detail: "Positive class is task-specific digit 1 or digit 8", execution: "plaintext" },
  );
  return stages;
}

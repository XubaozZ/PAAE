import os
import argparse
import numpy as np
from scipy import io as sio


def write_split(root, jpg_subdir, labels, split_ids, txt_name):
    lines = []
    missing = 0
    for img_id in sorted(split_ids.tolist()):
        rel = f"{jpg_subdir}/image_{img_id:05d}.jpg"
        if not os.path.exists(os.path.join(root, rel)):
            missing += 1
            continue
        target = int(labels[img_id - 1]) - 1
        lines.append(f"{rel} {target}")
    out = os.path.join(root, txt_name)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    msg = f"  wrote {out}: {len(lines)} lines"
    if missing:
        msg += f" ({missing} images missing)"
    print(msg)
    return len(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate train.txt / test.txt for Oxford 102 Flowers from imagelabels.mat and setid.mat. "
                    "The root must contain jpg/, imagelabels.mat and setid.mat. "
                    "Train = trnid + valid (2040 images), test = tstid (6149 images); "
                    "use --no-val to train on the official 1020-image split only. Labels are 0-indexed.")
    parser.add_argument("--root", type=str,
                        default=os.path.join(os.environ.get("PAAE_DATA_ROOT", "./data"), "flowers"),
                        help="Flowers root containing jpg/, imagelabels.mat, setid.mat "
                             "(default: $PAAE_DATA_ROOT/flowers)")
    parser.add_argument("--jpg-subdir", type=str, default="jpg",
                        help="Image subdirectory name under root (default: jpg)")
    parser.add_argument("--no-val", action="store_true",
                        help="Train on the official 1020-image train split only, excluding the validation split")
    args = parser.parse_args()

    img_dir = os.path.join(args.root, args.jpg_subdir)
    labels_mat = os.path.join(args.root, "imagelabels.mat")
    setid_mat = os.path.join(args.root, "setid.mat")
    for p in (img_dir, labels_mat, setid_mat):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}. Please place jpg/, imagelabels.mat and setid.mat under {args.root}.")

    labels = sio.loadmat(labels_mat)["labels"].reshape(-1).astype(int)
    setid = sio.loadmat(setid_mat)
    trn = setid["trnid"].reshape(-1).astype(int)
    val = setid["valid"].reshape(-1).astype(int)
    tst = setid["tstid"].reshape(-1).astype(int)

    train_ids = trn if args.no_val else np.concatenate([trn, val])
    test_ids = tst

    print(f"[flowers] number of classes = {len(np.unique(labels))} (expected 102)")
    n_tr = write_split(args.root, args.jpg_subdir, labels, train_ids, "train.txt")
    n_te = write_split(args.root, args.jpg_subdir, labels, test_ids, "test.txt")
    print(f"[flowers] train {n_tr} images / test {n_te} images (use_val_in_train={not args.no_val})")
    print("Done. You can now train with configs/swin-flowers.yaml.")


if __name__ == "__main__":
    main()

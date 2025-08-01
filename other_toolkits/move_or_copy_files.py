import os
import shutil
import argparse


def move_or_copy_files(src_dir, dest_dir, mode, cp_mode):
    images_dir = os.path.join(dest_dir, 'images', mode)
    labels_dir = os.path.join(dest_dir, 'labels', mode)

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)

        if os.path.isfile(src_file):
            if filename.endswith('.bmp'):
                dest_file = os.path.join(images_dir, filename)
                if cp_mode:
                    shutil.copy(src_file, dest_file)
                    print(f'Copied image: {filename} to {images_dir}')
                else:
                    shutil.move(src_file, dest_file)
                    print(f'Moved image: {filename} to {images_dir}')

            elif filename.endswith('.txt'):
                dest_file = os.path.join(labels_dir, filename)
                if cp_mode:
                    shutil.copy(src_file, dest_file)
                    print(f'Copied label: {filename} to {labels_dir}')
                else:
                    shutil.move(src_file, dest_file)
                    print(f'Moved label: {filename} to {labels_dir}')


def main():
    parser = argparse.ArgumentParser(description="Move or Copy .bmp images and .txt labels to specified subdirectories")
    parser.add_argument("source_dir", help="Path to the source directory containing .bmp and .txt files")
    parser.add_argument("--dest_dir", default="/root/data",
                        help="Path to the destination directory (default: /root/data)")
    parser.add_argument("--mode", default="train", choices=["train", "val", "test"],
                        help="The subdirectory mode (default: 'train')")
    parser.add_argument("--cp_mode", action="store_true",
                        help="Whether to copy files instead of moving them (default: False)")

    args = parser.parse_args()

    move_or_copy_files(args.source_dir, args.dest_dir, args.mode, args.cp_mode)


if __name__ == "__main__":
    main()

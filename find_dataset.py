
import os

base_path = r"C:\Users\Daneshwari\Downloads\archive"

for main_folder in os.listdir(base_path):

    main_path = os.path.join(base_path, main_folder)

    if os.path.isdir(main_path):

        print("\n================================")
        print("MAIN FOLDER:", main_folder)
        print("================================")

        for root, dirs, files in os.walk(main_path):

            level = root.replace(main_path, "").count(os.sep)

            if level > 3:
                dirs[:] = []
                continue

            indent = "    " * level
            print(indent + os.path.basename(root) + "/")

            image_files = [
                f for f in files
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp")
                )
            ]

            if image_files:
                print(indent + "    Images:", len(image_files))


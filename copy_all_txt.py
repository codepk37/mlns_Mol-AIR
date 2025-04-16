import os
import subprocess

# Base directory
base_dir = "/media/pavan/STORAGE/linux_storage/Gemini/Mol-AIR"
output_file = os.path.join(base_dir, "mol_air_now_SAC.txt")

# YAML files to add (with comments)
yaml_entries = [
    ("config/plogp/molair.yaml", "pLogP"),
    ("config/qed/molair.yaml", "QED"),
    ("config/similarity/molair.yaml", "Similarity"),
    ("config/gsk3b/molair.yaml", "GSK3B"),
    ("config/jnk3/molair.yaml", "JNK3"),
    ("config/gsk3b+jnk3/molair.yaml", "GSK3B+JNK3"),
]

with open(output_file, 'w', encoding='utf-8') as out_f:
    # Step 1: Run `tree` and capture output
    try:
        tree_output = subprocess.check_output(["tree", "."], cwd=base_dir, text=True)
        out_f.write("📁 DIRECTORY STRUCTURE (tree):\n")
        out_f.write(tree_output)
        out_f.write("\n" + "="*80 + "\n\n")
    except Exception as e:
        out_f.write(f"# Failed to run 'tree': {e}\n\n")

    # Step 2: Add all .py files
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, base_dir)

                out_f.write(f"{rel_path}\n")
                out_f.write(";;;\n")

                try:
                    with open(abs_path, 'r', encoding='utf-8') as in_f:
                        content = in_f.read()
                    out_f.write(content)
                except Exception as e:
                    out_f.write(f"# Error reading {rel_path}: {e}\n")

                out_f.write("\n;;;\n\n")

    # Step 3: Add selected YAML files with simple path + comment
    for rel_yaml_path, comment in yaml_entries:
        abs_yaml_path = os.path.join(base_dir, rel_yaml_path)

        out_f.write(f"{rel_yaml_path} # {comment}\n")
        out_f.write(";;;\n")

        try:
            with open(abs_yaml_path, 'r', encoding='utf-8') as yaml_f:
                yaml_content = yaml_f.read()
            out_f.write(yaml_content)
        except Exception as e:
            out_f.write(f"# Error reading {rel_yaml_path}: {e}\n")

        out_f.write("\n;;;\n\n")

print("✅ All content written to mol_air_now.txt")


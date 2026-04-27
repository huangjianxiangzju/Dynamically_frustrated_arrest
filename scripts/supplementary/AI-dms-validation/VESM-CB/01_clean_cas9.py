from pdbfixer import PDBFixer
from openmm.app import PDBFile
import os

# Your list of custom-named PDB files
pdb_files = [
    "7Z4C_6-nt.pdb", 
    "7Z4E_8-nt.pdb", 
    "7Z4G_12-nt.pdb", 
    "7Z4H_14-nt.pdb", 
    "7Z4I_16-nt.pdb", 
    "7Z4J_18cat-nt.pdb", 
    "7Z4K_10-nt.pdb", 
    "7Z4L_18-nt.pdb"
]

target_chain = 'B'

for filename in pdb_files:
    # Check if file exists to prevent hard crashes
    if not os.path.exists(filename):
        print(f"⚠️ Warning: {filename} not found in the current directory. Skipping.\n")
        continue

    try:
        print(f"⏳ Processing {filename}...")
        fixer = PDBFixer(filename=filename)
        
        # PDBFixer requires chain INDICES to remove, not the string IDs.
        # We find the index (i) of any chain that does not match 'A'
        chains_to_remove = [i for i, c in enumerate(fixer.topology.chains()) if c.id != target_chain]
        
        fixer.removeChains(chains_to_remove)
        
        # Repair the structure
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        
        # Create a clean output name (e.g., "7Z4C_6-nt_chainA_clean.pdb")
        base_name = filename.replace('.pdb', '')
        out_filename = f"{base_name}_chainA_clean.pdb"
        
        # Save the result
        with open(out_filename, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
            
        print(f"  ✓ Saved cleaned structure: {out_filename}\n")
        
    except Exception as e:
        print(f"  ❌ Error processing {filename}: {e}\n")

print("🎉 All files processed!")

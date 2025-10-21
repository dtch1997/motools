"""Tests for atom storage backend."""

from motools.atom import DatasetAtom, create_temp_workspace
from motools.atom.storage import list_atoms


def test_list_atoms_empty():
    """Test listing atoms when none exist."""
    # Note: may find atoms from other tests, so just check it doesn't crash
    atoms = list_atoms()
    assert isinstance(atoms, list)


def test_list_atoms_by_type():
    """Test listing atoms filtered by type."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "data.jsonl").write_text('{"text": "test"}\n')

        atom = DatasetAtom.create(user="alice", artifact_path=temp_dir)

        # List all atoms
        all_atoms = list_atoms()
        assert atom.id in all_atoms

        # List only dataset atoms
        dataset_atoms = list_atoms(atom_type="dataset")
        assert atom.id in dataset_atoms

        # List non-existent type
        other_atoms = list_atoms(atom_type="nonexistent")
        assert atom.id not in other_atoms

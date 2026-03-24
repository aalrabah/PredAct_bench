"""
PredAct Benchmark - Belief State Tracker
Initializes, validates, and manages the cumulative belief state across turns.
"""

import json
import copy
from config import ONTOLOGY_PATH


# =============================================================================
# LOAD ONTOLOGY
# =============================================================================

def load_ontology(path=None):
    """Load ontology.json which defines valid slot values."""
    path = path or ONTOLOGY_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# PARSE ONTOLOGY INTO DOMAIN STRUCTURE
# =============================================================================

def parse_ontology(ontology):
    """
    Parse flat ontology keys like "class_context-course_name"
    into nested domain structure:
    {
        "class_context": {
            "course_name": {"type": "categorical", "values": [...]},
            ...
        }
    }
    """
    domains = {}
    for key, values in ontology.items():
        parts = key.split("-", 1)
        if len(parts) != 2:
            continue
        domain, slot = parts

        if domain not in domains:
            domains[domain] = {}

        if values == "open_numeric":
            domains[domain][slot] = {"type": "open_numeric"}
        elif isinstance(values, list):
            domains[domain][slot] = {"type": "categorical", "values": values}
        else:
            domains[domain][slot] = {"type": "unknown"}

    return domains


# =============================================================================
# STATE TRACKER
# =============================================================================

class StateTracker:
    """
    Manages the cumulative belief state across dialogue turns.

    Responsibilities:
    - Initialize empty state from ontology
    - Validate slot values against ontology
    - Merge new values into current state (cumulative, never erase)
    - Track which slots were filled at which turn
    - Export state for metadata attachment
    """

    def __init__(self, ontology_path=None):
        self.ontology = load_ontology(ontology_path)
        self.schema = parse_ontology(self.ontology)
        self.state = self._init_empty_state()
        self.history = []  # list of (turn_number, state_snapshot)
        self.turn_count = 0

    def _init_empty_state(self):
        """Create empty belief state with all categorical slots set to ''."""
        state = {}
        for domain, slots in self.schema.items():
            if domain in ("student_status", "intervention"):
                # These use dynamic risk-group keys, start as empty dict
                state[domain] = {}
            else:
                state[domain] = {}
                for slot in slots:
                    state[domain][slot] = ""
        return state

    def reset(self):
        """Reset state for a new dialogue."""
        self.state = self._init_empty_state()
        self.history = []
        self.turn_count = 0

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_slot(self, domain, slot, value):
        """
        Validate a slot value against the ontology.
        Returns (is_valid, reason).
        """
        if domain not in self.schema:
            # Dynamic domains (student_status, intervention) skip validation
            # since they use risk-group keys not in the flat ontology
            return True, "dynamic_domain"

        if slot not in self.schema[domain]:
            return True, "dynamic_slot"

        slot_def = self.schema[domain][slot]

        if value == "" or value is None:
            return True, "empty"

        if slot_def["type"] == "open_numeric":
            if isinstance(value, (int, float)):
                return True, "valid_numeric"
            else:
                return False, f"expected numeric, got {type(value).__name__}"

        if slot_def["type"] == "categorical":
            if str(value) in slot_def["values"]:
                return True, "valid_categorical"
            else:
                return False, f"'{value}' not in {slot_def['values']}"

        return True, "unknown_type"

    def validate_state(self, state):
        """
        Validate an entire belief state dict.
        Returns list of (domain, slot, value, is_valid, reason).
        """
        results = []
        for domain, slots in state.items():
            if not isinstance(slots, dict):
                continue
            # For flat domains (class_context, class_summary, student_query)
            if domain in self.schema:
                for slot, value in slots.items():
                    is_valid, reason = self.validate_slot(domain, slot, value)
                    results.append((domain, slot, value, is_valid, reason))
            # For grouped domains (student_status, intervention)
            # We validate the inner fields against ontology where possible
            else:
                for group_key, group_data in slots.items():
                    if isinstance(group_data, dict):
                        for field, value in group_data.items():
                            # Try to validate against ontology
                            is_valid, reason = self.validate_slot(domain, field, value)
                            results.append((domain, f"{group_key}.{field}", value, is_valid, reason))
        return results

    # =========================================================================
    # UPDATE
    # =========================================================================

    def update(self, new_state):
        """
        Merge new belief state into current state.
        - Non-empty new values override current values
        - Empty/None values do NOT erase existing values (cumulative)
        - Grouped domains (student_status, intervention) merge at group level
        """
        if new_state is None:
            return self.state

        for domain, new_slots in new_state.items():
            if not isinstance(new_slots, dict):
                if new_slots != "" and new_slots is not None:
                    self.state[domain] = new_slots
                continue

            if domain not in self.state:
                self.state[domain] = {}

            current_domain = self.state[domain]

            for key, value in new_slots.items():
                if isinstance(value, dict):
                    # Nested dict (risk groups, per-student dicts)
                    if key not in current_domain:
                        current_domain[key] = {}
                    if isinstance(current_domain[key], dict):
                        for k, v in value.items():
                            if v != "" and v is not None:
                                current_domain[key][k] = v
                    else:
                        current_domain[key] = value
                else:
                    if value != "" and value is not None:
                        current_domain[key] = value

        # Record history
        self.turn_count += 1
        self.history.append((self.turn_count, copy.deepcopy(self.state)))

        return self.state

    # =========================================================================
    # QUERY
    # =========================================================================

    def get_state(self):
        """Return a deep copy of the current belief state."""
        return copy.deepcopy(self.state)

    def get_slot(self, domain, slot):
        """Get a specific slot value. Returns None if not found."""
        domain_data = self.state.get(domain, {})
        if isinstance(domain_data, dict):
            return domain_data.get(slot, None)
        return None

    def is_slot_filled(self, domain, slot):
        """Check if a slot has been filled (non-empty)."""
        value = self.get_slot(domain, slot)
        return value is not None and value != "" and value != {}

    def get_filled_slots(self):
        """Return all filled slots as a flat list of (domain, slot, value)."""
        filled = []
        for domain, slots in self.state.items():
            if isinstance(slots, dict):
                for slot, value in slots.items():
                    if value != "" and value is not None and value != {}:
                        filled.append((domain, slot, value))
        return filled

    def get_unfilled_slots(self):
        """Return all unfilled categorical slots."""
        unfilled = []
        for domain, slots in self.schema.items():
            if domain in ("student_status", "intervention"):
                if not self.state.get(domain):
                    unfilled.append((domain, "*", "empty_group"))
                continue
            for slot in slots:
                value = self.state.get(domain, {}).get(slot, "")
                if value == "" or value is None:
                    unfilled.append((domain, slot, ""))
        return unfilled

    def is_complete(self):
        """
        Check if the dialogue has reached a terminal state.
        Requires: class_context filled, class_summary filled,
                  student_status non-empty, intervention non-empty.
        """
        # class_context must have course_name and week
        if not self.is_slot_filled("class_context", "course_name"):
            return False
        if not self.is_slot_filled("class_context", "week"):
            return False

        # class_summary must have average_gpa and flagged_student_count
        if not self.is_slot_filled("class_summary", "average_gpa"):
            return False
        if not self.is_slot_filled("class_summary", "flagged_student_count"):
            return False

        # student_status must be non-empty
        if not self.state.get("student_status"):
            return False

        # intervention must be non-empty (or explicitly no intervention)
        if not self.state.get("intervention"):
            return False

        return True

    # =========================================================================
    # HISTORY
    # =========================================================================

    def get_history(self):
        """Return the full history of state snapshots."""
        return copy.deepcopy(self.history)

    def get_state_at_turn(self, turn_number):
        """Get the belief state as it was at a specific turn."""
        for t, state in self.history:
            if t == turn_number:
                return copy.deepcopy(state)
        return None

    def get_slot_fill_turn(self, domain, slot):
        """Find which turn a specific slot was first filled."""
        for t, state in self.history:
            domain_data = state.get(domain, {})
            if isinstance(domain_data, dict):
                value = domain_data.get(slot, "")
                if value != "" and value is not None:
                    return t
        return None

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_for_metadata(self):
        """Export current state formatted for dialogue metadata."""
        return copy.deepcopy(self.state)

    def export_summary(self):
        """Export a human-readable summary of the current state."""
        filled = self.get_filled_slots()
        unfilled = self.get_unfilled_slots()
        return {
            "turn": self.turn_count,
            "filled_count": len(filled),
            "unfilled_count": len(unfilled),
            "filled_slots": [(d, s) for d, s, _ in filled],
            "is_complete": self.is_complete(),
        }
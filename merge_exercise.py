# Merge Conflit 1: Edit same data from different branch
shared_data_structure = {}

def update_data_structure_developer_a(data_structure):
    data_structure['key_a'] = 'value_from_a'
    return data_structure

shared_data_structure = { 
    'key_a': 'initial_value_a',
    'key_b': 'initial_value_b',
    'common_key': 'common_initial_value'
}

def update_data_structure_developer_b(data_structure):
    data_structure['key_b'] = 'value_from_b'
    return data_structure

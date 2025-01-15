class AdvancedDFA:
    def __init__(self, states, alphabet, transition_function, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.start_state = start_state
        self.accept_states = accept_states

    def simulate(self, input_string):
        current_state = self.start_state
        print(f"Initial state: {current_state}")
        step = 1

        for symbol in input_string:
            if symbol not in self.alphabet:
                return f"Error: Symbol '{symbol}' not in alphabet"
            if current_state not in self.transition_function:
                return f"Error: No transition defined for state '{current_state}'"
            
            current_state = self.transition_function[current_state][symbol]
            print(f"Step {step}: Read '{symbol}', transitioned to state '{current_state}'")
            step += 1

        if current_state in self.accept_states:
            return f"Accepted: Ended in state '{current_state}'"
        else:
            return f"Rejected: Ended in state '{current_state}'"


# Advanced DFA configuration
states = {"q0", "q1", "q2", "q3"}
alphabet = {"0", "1"}
transition_function = {
    "q0": {"0": "q1", "1": "q2"},
    "q1": {"0": "q0", "1": "q3"},
    "q2": {"0": "q3", "1": "q2"},
    "q3": {"0": "q3", "1": "q3"}
}
start_state = "q0"
accept_states = {"q3"}

# Simulate DFA with more complex input
dfa = AdvancedDFA(states, alphabet, transition_function, start_state, accept_states)
input_string = "001101010"  # Complex hardcoded input
print(f"Input: {input_string}")
result = dfa.simulate(input_string)
print(f"Result: {result}")

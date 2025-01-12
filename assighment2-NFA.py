class AdvancedNFA:
    def __init__(self, states, alphabet, transition_function, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.start_state = start_state
        self.accept_states = accept_states

    def simulate(self, input_string):
        current_states = {self.start_state}
        print(f"Initial states: {current_states}")
        step = 1

        for symbol in input_string:
            if symbol not in self.alphabet:
                return f"Error: Symbol '{symbol}' not in alphabet"
            next_states = set()
            for state in current_states:
                if state in self.transition_function and symbol in self.transition_function[state]:
                    next_states.update(self.transition_function[state][symbol])
            current_states = next_states
            print(f"Step {step}: Read '{symbol}', current states: {current_states}")
            step += 1

        if any(state in self.accept_states for state in current_states):
            return f"Accepted: Reached accept state(s) {current_states & self.accept_states}"
        else:
            return f"Rejected: Current states {current_states} do not include an accept state"


# Advanced NFA configuration
states = {"q0", "q1", "q2", "q3", "q4"}
alphabet = {"0", "1"}
transition_function = {
    "q0": {"0": ["q0", "q1"], "1": ["q0"]},
    "q1": {"0": ["q2"], "1": ["q3"]},
    "q2": {"0": ["q4"]},
    "q3": {"1": ["q4"]},
    "q4": {"0": ["q4"], "1": ["q4"]}
}
start_state = "q0"
accept_states = {"q4"}

# Simulate NFA with more complex input
nfa = AdvancedNFA(states, alphabet, transition_function, start_state, accept_states)
input_string = "001101010"  # Complex hardcoded input
print(f"Input: {input_string}")
result = nfa.simulate(input_string)
print(f"Result: {result}")

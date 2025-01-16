class AdvancedNFA:
    def __init__(self, states, alphabet, transition_function, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.start_state = start_state
        self.accept_states = accept_states

    def simulate(self, input_string):
        current_states = {self.start_state}
        print(f"Starting NFA Simulation")
        print(f"States: {self.states}")
        print(f"Alphabet: {self.alphabet}")
        print(f"Start State: {self.start_state}")
        print(f"Accept States: {self.accept_states}")
        print(f"Transition Function: {self.transition_function}")
        print(f"Input String: {input_string}")
        print(f"Initial States: {current_states}\n")

        step = 1

        for symbol in input_string:
            print(f"Step {step}:")
            print(f"Current States: {current_states}")
            print(f"Reading Symbol: '{symbol}'")

            if symbol not in self.alphabet:
                return f"Error: Symbol '{symbol}' not in alphabet. Terminating simulation."

            next_states = set()
            for state in current_states:
                if state in self.transition_function and symbol in self.transition_function[state]:
                    next_states.update(self.transition_function[state][symbol])

            print(f"Transition: States {current_states} --'{symbol}'--> {next_states}")
            current_states = next_states
            step += 1
            print(f"States After Transition: {current_states}\n")

        print("Simulation Complete!")
        print(f"Final States: {current_states}")

        if any(state in self.accept_states for state in current_states):
            return f"Result: Accepted. Reached accept state(s) {current_states & self.accept_states}."
        else:
            return f"Result: Rejected. None of the current states {current_states} are accept states."

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
print(f"\n{result}")

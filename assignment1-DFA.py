class AdvancedDFA:
    def __init__(self, states, alphabet, transition_function, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.start_state = start_state
        self.accept_states = accept_states

    def simulate(self, input_string):
        current_state = self.start_state
        print(f"Starting DFA Simulation")
        print(f"States: {self.states}")
        print(f"Alphabet: {self.alphabet}")
        print(f"Start State: {self.start_state}")
        print(f"Accept States: {self.accept_states}")
        print(f"Transition Function: {self.transition_function}")
        print(f"Input String: {input_string}")
        print(f"Initial State: {current_state}\n")
        
        step = 1

        for symbol in input_string:
            print(f"Step {step}:")
            print(f"Current State: {current_state}")
            print(f"Reading Symbol: '{symbol}'")

            if symbol not in self.alphabet:
                return f"Error: Symbol '{symbol}' not in alphabet. Terminating simulation."

            if current_state not in self.transition_function:
                return f"Error: No transition defined for state '{current_state}'. Terminating simulation."
            
            next_state = self.transition_function[current_state][symbol]
            print(f"Transition: {current_state} --'{symbol}'--> {next_state}")
            current_state = next_state
            step += 1
            print(f"State After Transition: {current_state}\n")

        print("Simulation Complete!")
        print(f"Final State: {current_state}")

        if current_state in self.accept_states:
            return f"Result: Accepted. The input string ended in an accept state '{current_state}'."
        else:
            return f"Result: Rejected. The input string ended in a non-accept state '{current_state}'."

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
print(f"\n{result}")

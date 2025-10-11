"""
OpenSpiel environment wrapper for Stud Poker.
"""
import numpy as np
import pyspiel


class StudPokerEnv:
    """
    Wrapper for OpenSpiel Stud Poker environment.
    Provides a standard interface for RL agents.
    """
    
    def __init__(self, game_name="kuhn_poker", num_players=2):
        """
        Initialize the poker environment.
        
        Args:
            game_name: Name of the poker game (default: kuhn_poker)
            num_players: Number of players in the game
        """
        self.game_name = game_name
        self.num_players = num_players
        self.game = pyspiel.load_game(game_name)
        self.state = None
        self.reset()
        
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        self.state = self.game.new_initial_state()
        return self._get_observation()
    
    def _get_observation(self):
        """
        Get the current observation from the state.
        
        Returns:
            Observation as numpy array
        """
        if self.state.is_terminal():
            return None
        
        # Get information state tensor for current player
        current_player = self.state.current_player()
        if current_player == pyspiel.PlayerId.CHANCE:
            return None
        
        info_state = self.state.information_state_tensor(current_player)
        return np.array(info_state, dtype=np.float32)
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: Action to take
        
        Returns:
            observation: Next observation
            reward: Reward received
            done: Whether episode is finished
            info: Additional information
        """
        if self.state.is_terminal():
            raise ValueError("Cannot step in terminal state. Call reset().")
        
        current_player = self.state.current_player()
        
        # Handle chance nodes
        while current_player == pyspiel.PlayerId.CHANCE:
            outcomes = self.state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            self.state.apply_action(action)
            if self.state.is_terminal():
                break
            current_player = self.state.current_player()
        
        if not self.state.is_terminal():
            self.state.apply_action(action)
        
        # Get reward and next state
        done = self.state.is_terminal()
        reward = 0.0
        if done:
            returns = self.state.returns()
            reward = returns[current_player]
        
        observation = self._get_observation()
        info = {
            'legal_actions': self.get_legal_actions() if not done else [],
            'current_player': self.state.current_player() if not done else None
        }
        
        return observation, reward, done, info
    
    def get_legal_actions(self):
        """
        Get list of legal actions in current state.
        
        Returns:
            List of legal action indices
        """
        if self.state.is_terminal():
            return []
        
        if self.state.current_player() == pyspiel.PlayerId.CHANCE:
            return []
        
        return self.state.legal_actions()
    
    def get_observation_shape(self):
        """
        Get the shape of observations.
        
        Returns:
            Shape tuple
        """
        return (len(self.state.information_state_tensor(0)),)
    
    def get_action_space_size(self):
        """
        Get the size of the action space.
        
        Returns:
            Number of possible actions
        """
        return self.game.num_distinct_actions()
    
    def render(self):
        """
        Render the current state.
        """
        print(self.state)

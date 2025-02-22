from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
import numpy.typing as npt
from typing import List, Tuple, Optional

class Firm:
    """
    Data class to store firm information
    """
    def __init__(self,
                 position: npt.NDArray[np.float64],
                 price: float):
        self.position = position
        self.price = price
        self.index = 0

    def __repr__(self) -> str:
        """String representation of Q structure."""
        return f"Firm({self.position}, {self.price})"

class SpatialCompetitionModel(ABC):
    def __init__(self, 
                 N: int,
                 beta: float, 
                 cost: float,
                 manifold_volume: float,
                 pi: float,
                 firms: Optional[List[Firm]] = None):
        """
        Initialize the abstract spatial competition model.
        
        Args:
            beta: Smoothing parameter for softmin
            cost: Cost parameter for all firms
            manifold_volume: Volume of the manifold
            pi: Intensity of the Poisson Point Process
            firms: List of firms (positions and prices will be initialized later)
        """
        self.num_firms = N
        self.beta = beta
        self.cost = cost
        self.manifold_volume = manifold_volume
        self.pi = pi
        if firms == None:
            self.firms = self.get_initial_firms()
        else:
            self.firms = firms

        for i, firm in enumerate(self.firms):
            firm.index = i
    
    @abstractmethod
    def distance_manifold(self, 
                         x: npt.NDArray[np.float64], 
                         y: npt.NDArray[np.float64]) -> float:
        """
        Abstract method to compute distance on the manifold between two points.
        Must be implemented by concrete subclasses for specific manifolds.
        """
        pass
    
    @abstractmethod
    def generate_integration_points(self) -> Tuple[npt.NDArray[np.float64], float]:
        """
        Abstract method to generate points for numerical integration over the manifold.
        Returns:
            Tuple of (points, weight_per_point)
        Must be implemented by concrete subclasses.
        """
        pass
    
    def market_share(self, 
                    firm: Firm,
                    x: npt.NDArray[np.float64]) -> float:
        """
        Compute market share at point x using softmin formula.
        """
        distances_prices = np.array([
            self.distance_manifold(x, f.position) + f.price 
            for f in self.firms
        ])
        exp_terms = np.exp(-self.beta * distances_prices)
        firm_index = self.firms.index(firm)
        
        return exp_terms[firm_index] / np.sum(exp_terms, axis = 0)
    
    def profit(self, 
              firm: Firm) -> float:
        """
        Compute total profit for a given firm.
        """
        integration_points, weight = self.generate_integration_points()
        
        total_market_share = sum(
            self.market_share(firm, point)
            for point in integration_points
        )
        
        # Scale by integration weight and PPP intensity
        total_market_share *= weight * self.pi / self.manifold_volume
        
        return (firm.price - self.cost) * total_market_share
    
    def best_response(self, 
                     firm_index: int,
                     initial_guess: Tuple[npt.NDArray[np.float64], float] | None = None
                     ) -> Tuple[npt.NDArray[np.float64], float]:
        """
        Compute the best response for a given firm given other firms' strategies.
        """
        if initial_guess is None:
            # Use current position and price as initial guess
            position_guess = self.firms[firm_index].position
            price_guess = self.firms[firm_index].price
            initial_guess = (position_guess, price_guess)

        old_position = self.firms[firm_index].position
        old_price = self.firms[firm_index].price
            
        def negative_profit(x):
            # Temporarily update firm's strategy
            self.firms[firm_index].position = x[:-1]
            self.firms[firm_index].price = x[-1]
            
            profit = self.profit(self.firms[firm_index])
            
            # Restore original strategy
            self.firms[firm_index].position = old_position
            self.firms[firm_index].price = old_price
            
            return -profit
        
        x0 = np.concatenate([initial_guess[0], [initial_guess[1]]])
        bounds = self.get_optimization_bounds()
        
        result = minimize(negative_profit, x0, bounds=bounds)
        return result.x[:-1], result.x[-1]

    def best_response_price(self, 
                          firm_index: int,
                          initial_price: float | None = None) -> float:
        """
        Compute the best response price for a given firm while keeping position fixed.
        
        Args:
            firm_index: Index of the firm to optimize
            initial_price: Initial guess for price optimization. If None, uses current price.
            
        Returns:
            Optimal price for the firm
        """
        if initial_price is None:
            initial_price = self.firms[firm_index].price
            
        def negative_profit(price):
            old_price = self.firms[firm_index].price
            
            # Temporarily update firm's price
            self.firms[firm_index].price = price[0]
            
            profit = self.profit(self.firms[firm_index])
            
            # Restore original price
            self.firms[firm_index].price = old_price
            
            return -profit
        
        # Get price bounds from the full optimization bounds
        price_bounds = [self.get_optimization_bounds()[-1]]
        
        result = minimize(negative_profit, [initial_price], bounds=price_bounds)
        return result.x[0]

    def best_response_position(self, 
                             firm_index: int,
                             initial_position: npt.NDArray[np.float64] | None = None
                             ) -> npt.NDArray[np.float64]:
        """
        Compute the best response position for a given firm while keeping price fixed.
        
        Args:
            firm_index: Index of the firm to optimize
            initial_position: Initial guess for position optimization. If None, uses current position.
            
        Returns:
            Optimal position for the firm
        """
        if initial_position is None:
            initial_position = self.firms[firm_index].position
            
        def negative_profit(position):
            old_position = self.firms[firm_index].position
            
            # Temporarily update firm's position
            self.firms[firm_index].position = position
            
            profit = self.profit(self.firms[firm_index])
            
            # Restore original position
            self.firms[firm_index].position = old_position
            
            return -profit
        
        # Get position bounds from the full optimization bounds (excluding price bound)
        position_bounds = self.get_optimization_bounds()[:-1]
        
        result = minimize(negative_profit, initial_position, bounds=position_bounds)
        return result.x
    
    @abstractmethod
    def get_optimization_bounds(self) -> List[Tuple[float, float]]:
        """
        Abstract method to get bounds for optimization.
        Must be implemented by concrete subclasses based on manifold structure.
        """
        pass
    
    def find_nash_equilibrium(self,
                            tolerance: float = 1e-4, 
                            max_iterations: int = 1000) -> None:
        """
        Find Nash equilibrium using best response iteration.
        Updates firms' positions and prices in place.
        """
        for _ in range(max_iterations):
            old_positions = [f.position.copy() for f in self.firms]
            old_prices = [f.price for f in self.firms]
            
            # Update each firm's strategy
            for i in range(self.num_firms):
                new_position, new_price = self.best_response(i)
                self.firms[i].position = new_position
                self.firms[i].price = new_price
            
            # Check convergence
            if all(np.linalg.norm(f.position - old_pos) < tolerance and
                   abs(f.price - old_price) < tolerance
                   for f, old_pos, old_price in zip(self.firms, old_positions, old_prices)):
                break
    
    @abstractmethod
    def get_initial_firms(self) -> List[Firm]:
        """
        Abstract method to get initial positions and prices for firms.
        Must be implemented by concrete subclasses based on manifold structure.
        """
        pass
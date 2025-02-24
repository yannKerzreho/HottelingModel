import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from Class import SpatialCompetitionModel, Firm
from scipy.optimize import minimize

class CircularModel(SpatialCompetitionModel):
    def __init__(self,
                 N: int,
                 beta: float,
                 cost: float,
                 pi: float,
                 firms: Optional[List[Firm]] = None):
        """
        Initialize Hotelling model on unit circle [0, 2π].
        
        Args:
            N: Number of firms
            beta: Smoothing parameter for softmin
            cost: Cost parameter for all firms
            pi: Intensity of the Poisson Point Process
            firms: Optional list of firms (if None, will be initialized)
        """
        super().__init__(
            N=N,
            beta=beta,
            cost=cost,
            manifold_volume=2 * np.pi,  # Circle circumference is 2π
            pi=pi,
            firms=firms
        )

    def distance_manifold(self,
                         x: npt.NDArray[np.float64],
                         y: npt.NDArray[np.float64]) -> float:
        """
        Compute distance on unit circle.
        Uses minimum of direct distance and distance going the other way around the circle.
        """
        direct_distance = np.abs(x - y)
        return np.minimum(direct_distance, 2 * np.pi - direct_distance)

    def generate_integration_points(self) -> Tuple[npt.NDArray[np.float64], float]:
        """
        Generate 1000 equally spaced points on [0, 2π] for numerical integration.
        
        Returns:
            Tuple of (points array, weight per point)
        """
        num_points = 1000
        points = np.linspace(0, 2 * np.pi, num_points)
        # Reshape points to be compatible with ndarray operations
        points = points.reshape(-1, 1)
        # Weight is circle circumference divided by number of points
        weight = (2 * np.pi) / num_points
        
        return points, weight

    def get_optimization_bounds(self) -> List[Tuple[float, float]]:
        """
        Get bounds for optimization.
        Position must be in [0, 2π], price must be above cost.
        """
        return [(0, 2 * np.pi),    # position bound
                (self.cost, None)]  # price bound

    def get_initial_firms(self) -> List[Firm]:
        """
        Get initial positions and prices for firms.
        Positions are equally spaced on [0, 2π], prices start at 2*cost.
        """
        firms = []
        for i in range(self.num_firms):
            # Spread firms equally on circle
            position = np.array([2 * np.pi * (i + 1) / self.num_firms])
            # Initial price is twice the cost
            price = 2 * self.cost
            firms.append(Firm(position=position, price=price))
        
        return firms
    
    def get_unifeq_price(self, firm_index) -> float:

        x, weight = self.generate_integration_points()

        distances_prices = np.array([
            self.distance_manifold(x, f.position) + f.price 
            for f in self.firms
        ])

        exp_terms = np.exp(-self.beta * distances_prices)

        sumexp = np.sum(exp_terms, axis = 0)
        fi = exp_terms[0] / sumexp
        onefi = (1-exp_terms[0]) / sumexp
        
        return np.sum(fi)/ (self.beta * sum(fi * onefi))

    def get_intial_points(self, firm_index) -> List[Tuple[npt.NDArray[np.float64], float]]:
        return [
            (self.firms[firm_index].position, self.firms[firm_index].price),
            # Position at π with firm's price
            (np.array([np.pi]), self.firms[firm_index].price),
            # Position at 0 with firm's price
            (np.array([0.0]), self.firms[firm_index].price)
        ]
    
    def __repr__(self) -> str:
        """String representation of CircularModel structure."""
        header = f"CircularModel(beta={self.beta}, cost={self.cost}\n"
        firms_str = "\nFirms:\n"
        for i, firm in enumerate(self.firms):
            # Convert position to radians for clearer representation
            pos_rad = firm.position[0]
            pos_deg = np.degrees(pos_rad)
            firms_str += f"  {i+1}: position={pos_rad:.3f} rad ({pos_deg:.1f}°), price={firm.price:.3f}\n"
        return header + firms_str + ")"

    def visualize_market_shares(self, num_points: int = 1000):
        """
        Visualize market shares along the circle [0, 2π].
        """        
        x = np.linspace(0, 2 * np.pi, num_points).reshape(-1, 1)
        shares = np.zeros((len(self.firms), num_points))
        
        for i, firm in enumerate(self.firms):
            shares[i, :] = [self.market_share(firm, x_i) for x_i in x]
        
        plt.figure(figsize=(12, 6))
        for i in range(len(self.firms)):
            plt.plot(x, shares[i, :], label=f'Firm {i+1}')
            plt.axvline(x=self.firms[i].position, color=f'C{i}', 
                       linestyle='--', alpha=0.5)
        
        plt.xlabel('Position (radians)')
        plt.ylabel('Market Share')
        plt.title('Market Shares and Firm Positions')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.linspace(0, 2 * np.pi, 7), 
                  ['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'])
        plt.show()

    def plot_profit_curve(self, 
                        firm_index: int,
                        vary_price: bool = True,
                        num_points: int = 100,
                        position_range: Optional[Tuple[float, float]] = None,
                        price_range: Optional[Tuple[float, float]] = None):
        """
        Plot profit curve when varying either price or position while keeping the other fixed.
        For positions, works in the circular domain [0, 2π].
        
        Args:
            firm_index: Index of the firm to analyze
            vary_price: If True, varies price and keeps position fixed. If False, varies position
            num_points: Number of points to evaluate
            position_range: Optional tuple of (min_pos, max_pos). If None, uses [0, 2π]
            price_range: Optional tuple of (min_price, max_price). If None, uses [cost, 3*cost]
        """
        # Set up ranges
        if position_range is None:
            position_range = (0, 2 * np.pi)  # Full circle
        if price_range is None:
            price_range = (self.cost, 2 * self.cost)
        
        # Store original values
        current_profit = float(self.profit(self.firms[firm_index]))
        original_position = self.firms[firm_index].position.copy()
        original_price = self.firms[firm_index].price
        
        # Create array of values to test
        if vary_price:
            x_values = np.linspace(price_range[0], price_range[1], num_points)
        else:
            x_values = np.linspace(position_range[0], position_range[1], num_points)
        
        # Calculate profits
        profits = np.zeros(num_points)
        
        for i, x in enumerate(x_values):
            if vary_price:
                self.firms[firm_index].price = float(x)
            else:
                self.firms[firm_index].position = np.array([float(x)])
            profits[i] = float(self.profit(self.firms[firm_index]))
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot profit curve
        plt.plot(x_values, profits, 'b-', label='Profit')
        
        # Plot current position/price
        if vary_price:
            current_x = original_price
        else:
            current_x = original_position
            
        plt.plot(current_x, current_profit, 'ko', markersize=10,
                label=f'Current Position (profit: {current_profit:.3f})')
        
        # Add labels and title
        plt.xlabel('Price' if vary_price else 'Position (radians)')
        plt.ylabel('Profit')
        plt.title(f'Profit vs {"Price" if vary_price else "Position"} for Firm {firm_index}')
        plt.grid(True)
        
        # Special x-axis formatting for positions
        if not vary_price:
            # Set x-ticks to show π fractions
            plt.xticks(np.linspace(0, 2 * np.pi, 7),
                    ['0', 'π/3', '2π/3', 'π', '4π/3', '5π/3', '2π'])
        
        plt.legend()
        
        # Restore original values
        self.firms[firm_index].position = original_position
        self.firms[firm_index].price = original_price
        
        plt.show()
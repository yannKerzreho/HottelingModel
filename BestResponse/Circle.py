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
        num_points = 10000
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
            firms.append(Firm(position=position, price=2*self.cost))
        
        return firms
    
    def get_unifeq_price(self, precision = 1e-4) -> float:

        firms = []
        price = 2*self.cost
        for i in range(self.num_firms):
            position = np.array([2 * np.pi * (i + 1) / self.num_firms])
            firms.append(Firm(position=position, price=price))

        self.firms = firms

        price = self.cost * 2

        for _ in range(1000):
            old_price = price
            price = self.best_response_price(0)
            print(f'Debug: price={price} | old price={old_price} | abs={abs(old_price - price)}')
            for i in range(self.num_firms):
                self.firms[i].price = price
            if abs(price - old_price) < precision:
                return price

        return price

    def get_secondOrderDrivative_Dmd_pos(self, firm_index) -> float:

        x, w = self.generate_integration_points()

        pos_i = self.firms[firm_index].position
        pos_bari = (pos_i + np.pi) % (2 * np.pi)

        # Compute fi
        distances_prices = np.array([
            self.distance_manifold(x, f.position) + f.price 
            for f in self.firms
        ])
        exp_terms = np.exp(-self.beta * distances_prices)        
        fi = exp_terms[firm_index] / np.sum(exp_terms, axis = 0)
        fi = fi / self.manifold_volume

        # Compute fi at theta_i
        distances_prices_theta_i = np.array([
            self.distance_manifold(pos_i, f.position) + f.price 
            for f in self.firms
        ])
        exp_terms_theta_i = np.exp(-self.beta * distances_prices_theta_i)        
        fi_theta_i = exp_terms_theta_i[firm_index] / np.sum(exp_terms_theta_i, axis = 0)
        fi_theta_i = fi_theta_i / self.manifold_volume

        # Compute fi at bar_theta_i
        distances_prices_bartheta_i = np.array([
            self.distance_manifold(pos_bari, f.position) + f.price 
            for f in self.firms
        ])
        exp_terms_bartheta_i = np.exp(-self.beta * distances_prices_bartheta_i)        
        fi_bartheta_i = exp_terms_bartheta_i[firm_index] / np.sum(exp_terms_bartheta_i, axis = 0)
        fi_bartheta_i = fi_bartheta_i / self.manifold_volume

        # Compute int +\beta^2
        int1 = self.beta**2 * np.sum(fi * (1-2*fi)* (1-fi)* w)
        # Compute -\beta fi (1-fi) at \theta i
        int2 = - 2 * self.beta * fi_theta_i * (1-fi_theta_i)
        # Compute \beta fi (1-fi) at \bar \theta i
        int3 = 2 * self.beta * fi_bartheta_i * (1-fi_bartheta_i)

        return int1 + int2 + int3
    
    def get_concetraed_price(self, precision = 1e-4) -> float:

        firms = []
        price = 2*self.cost
        for i in range(self.num_firms):
            firms.append(Firm(position=np.array([0]), price=price))

        self.firms = firms

        price = self.cost * 2

        for _ in range(1000):
            old_price = price
            price = self.best_response_price(0)
            print(f'Debug: price={price} | old price={old_price} | abs={abs(old_price - price)}')
            for i in range(self.num_firms):
                self.firms[i].price = price
            if abs(price - old_price) < precision:
                return price

        return price

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
            price_range = (self.cost, 5 * self.cost)
        
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
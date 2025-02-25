import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from Class import SpatialCompetitionModel, Firm
from scipy.optimize import minimize

class LinearModel(SpatialCompetitionModel):
    def __init__(self,
                 N: int,
                 beta: float,
                 cost: float,
                 pi: float,
                 firms: Optional[List[Firm]] = None):
        """
        Initialize Hotelling model on [0,1] segment.
        
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
            manifold_volume=1.0,  # [0,1] segment has volume 1
            pi=pi,
            firms=firms
        )

    def distance_manifold(self,
                         x: npt.NDArray[np.float64],
                         y: npt.NDArray[np.float64]) -> float:
        """
        Compute L1 distance on [0,1] segment.
        For 1D case, this is just the absolute difference.
        """
        return np.abs(x - y)

    def generate_integration_points(self) -> Tuple[npt.NDArray[np.float64], float]:
        """
        Generate 1000 equally spaced points on [0,1] for numerical integration.
        
        Returns:
            Tuple of (points array, weight per point)
        """
        num_points = 10000
        points = np.linspace(0, 1, num_points)
        # Reshape points to be compatible with ndarray operations
        points = points.reshape(-1, 1)
        # Weight is segment length divided by number of points
        weight = 1.0 / num_points
        
        return points, weight

    def get_optimization_bounds(self) -> List[Tuple[float, float]]:
        """
        Get bounds for optimization.
        Position must be in [0,1], price must be above cost.
        """
        return [(0, 1),          # position bound
                (self.cost, None)]  # price bound

    def get_initial_firms(self) -> List[Firm]:
        """
        Get initial positions and prices for firms.
        Positions are equally spaced on [0,1], prices start at 2*cost.
        """
        firms = []
        for i in range(self.num_firms):
            # Spread firms equally on [0,1]
            position = np.array([(i + 1) / (self.num_firms + 1)])
            # Initial price is twice the cost
            price = 2 * self.cost
            firms.append(Firm(position=position, price=price))
        
        return firms
    
    def get_intial_points(self, firm_index) -> List[Tuple[npt.NDArray[np.float64], float]]:
        """
        Methode to generate the intial points of the optimisation probleme from best response.
        Returns:
            List of Tuple of (points, weight_per_point).
        """
        return  [
            (self.firms[firm_index].position, self.firms[firm_index].price),
            (np.ones_like(self.firms[firm_index].position), self.firms[firm_index].price)
        ]
    
    def __repr__(self) -> str:
        """String representation of LinearModel structure."""
        header = f"LinearModel(beta={self.beta}, cost={self.cost}\n"
        firms_str = "\nFirms:\n"
        for i, firm in enumerate(self.firms):
            firms_str += f"  {i+1}: position={firm.position[0]:.3f}, price={firm.price:.3f}\n"
        return header + firms_str + ")"

    def visualize_market_shares(self, num_points: int = 1000):
        """
        Visualize market shares along the [0,1] segment.
        """        
        x = np.linspace(0, 1, num_points).reshape(-1, 1)
        shares = np.zeros((len(self.firms), num_points))
        
        for i, firm in enumerate(self.firms):
            shares[i, :] = [self.market_share(firm, x_i) for x_i in x]
        
        plt.figure(figsize=(12, 6))
        for i in range(len(self.firms)):
            plt.plot(x, shares[i, :], label=f'Firm {i+1}')
            plt.axvline(x=self.firms[i].position, color=f'C{i}', 
                       linestyle='--', alpha=0.5)
        
        plt.xlabel('Position on [0,1]')
        plt.ylabel('Market Share')
        plt.title('Market Shares and Firm Positions')
        plt.legend()
        plt.grid(True)
        plt.show()

    def response_graph(self, 
                    firm_index: int,
                    num_points: int = 10,
                    position_range: Optional[Tuple[float, float]] = None,
                    price_range: Optional[Tuple[float, float]] = None):
        """
        Create a 2D intensity plot showing profit landscape for different positions and prices.
        Compare with best response solution.
        
        Args:
            firm_index: Index of the firm to analyze
            num_points: Number of points to evaluate in each dimension
            position_range: Optional tuple of (min_pos, max_pos). If None, uses [0,1]
            price_range: Optional tuple of (min_price, max_price). If None, uses [cost, 3*cost]
        """        
        # Set up ranges for position and price
        if position_range is None:
            position_range = (0, 1)
        if price_range is None:
            price_range = (self.cost, 2 * self.cost)
        
        # Create meshgrid for positions and prices
        positions = np.linspace(position_range[0], position_range[1], num_points)
        prices = np.linspace(price_range[0], price_range[1], num_points)
        pos_mesh, price_mesh = np.meshgrid(positions, prices)
        
        # Calculate profit for each position-price combination
        profits = np.zeros((num_points, num_points))
        
        original_position = self.firms[firm_index].position.copy()
        original_price = self.firms[firm_index].price
        
        # Store the maximum profit and its location
        max_profit = float('-inf')
        max_pos = None
        max_price = None
        
        for i in range(num_points):
            for j in range(num_points):
                self.firms[firm_index].position = np.array([pos_mesh[i,j]])
                self.firms[firm_index].price = price_mesh[i,j]
                profit = self.profit(self.firms[firm_index])
                profits[i,j] = profit
                
                if profit > max_profit:
                    max_profit = profit
                    max_pos = float(pos_mesh[i,j])  # Convert to float
                    max_price = float(price_mesh[i,j])  # Convert to float
        
        # Restore original position and price
        self.firms[firm_index].position = original_position
        self.firms[firm_index].price = original_price
        
        # Calculate best response
        br_position, br_price = self.best_response(firm_index)
        self.firms[firm_index].position = br_position
        self.firms[firm_index].price = br_price
        br_profit = self.profit(self.firms[firm_index])
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Create intensity plot
        im = plt.imshow(profits, 
                        extent=[position_range[0], position_range[1], 
                            price_range[0], price_range[1]],
                        origin='lower',
                        aspect='auto',
                        cmap='viridis')
        
        # Add colorbar
        plt.colorbar(im, label='Profit')
        
        # Plot maximum point from grid search
        plt.plot(max_pos, max_price, 'r*', markersize=15, 
                label=f'Grid Max (profit: {max_profit})')
        
        # Plot best response solution
        plt.plot(br_position[0], br_price, 'w*', markersize=15, 
                label=f'Best Response (profit: {br_profit})')
        

        # Restore original position and price again
        self.firms[firm_index].position = original_position
        self.firms[firm_index].price = original_price
        
        # Plot current positions of all firms
        for i, firm in enumerate(self.firms):
            if i == firm_index:
                plt.plot(float(firm.position[0]), firm.price, 'ro', 
                        label='Current Position')
            else:
                plt.plot(float(firm.position[0]), firm.price, 'ko', 
                        label='Other Firms' if i == 0 else "")
        
        plt.xlabel('Position')
        plt.ylabel('Price')
        plt.title(f'Profit Landscape for Firm {firm_index + 1}')
        plt.legend()
        plt.grid(False)
        plt.show()
        
        return {
            'grid_search': {
                'position': max_pos,
                'price': max_price,
                'profit': max_profit
            },
            'best_response': {
                'position': float(br_position[0]),
                'price': br_price,
                'profit': br_profit
            }
        }
    
    def plot_profit_curve(self, 
                        firm_index: int,
                        vary_price: bool = True,
                        num_points: int = 100,
                        position_range: Optional[Tuple[float, float]] = None,
                        price_range: Optional[Tuple[float, float]] = None):
        """
        Plot profit curve when varying either price or position while keeping the other fixed.
        
        Args:
            firm_index: Index of the firm to analyze
            vary_price: If True, varies price and keeps position fixed. If False, varies position
            num_points: Number of points to evaluate
            position_range: Optional tuple of (min_pos, max_pos). If None, uses [0,1]
            price_range: Optional tuple of (min_price, max_price). If None, uses [cost, 3*cost]
        """
        # Set up ranges
        if position_range is None:
            position_range = (0, 1)
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
        
        # Calculate best response
        if vary_price:
            best_response = float(self.best_response_price(firm_index))
            self.firms[firm_index].price = best_response
        else:
            best_response = self.best_response_position(firm_index)
            self.firms[firm_index].position = best_response
        
        best_response_profit = float(self.profit(self.firms[firm_index]))
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot profit curve
        plt.plot(x_values, profits, 'b-', label='Profit')
        
        # Plot best response point
        if vary_price:
            br_x = best_response
        else:
            br_x = float(best_response[0])
        
        plt.plot(br_x, best_response_profit, 'r*', markersize=15,
                label=f'Best Response (profit: {float(best_response_profit):.3f})')
        
        # Plot current position/price
        if vary_price:
            current_x = float(original_price)
        else:
            current_x = float(original_position[0])
            
        plt.plot(current_x, current_profit, 'ko', markersize=10,
                label=f'Current Position (profit: {current_profit:.3f})')
        
        # Add labels and title
        plt.xlabel('Price' if vary_price else 'Position')
        plt.ylabel('Profit')
        plt.title(f'Profit vs {"Price" if vary_price else "Position"} for Firm {firm_index + 1}')
        plt.grid(True)
        plt.legend()
        
        # Restore original values
        self.firms[firm_index].position = original_position
        self.firms[firm_index].price = original_price
        
        plt.show()
        
        return {
            'best_response': {
                'value': float(br_x),
                'profit': float(best_response_profit)
            },
            'current': {
                'value': float(current_x),
                'profit': float(current_profit)
            }
        }
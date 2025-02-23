import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional
from Class import SpatialCompetitionModel, Firm
from scipy.optimize import minimize


class CubeModel(SpatialCompetitionModel):
    def __init__(self,
                 N: int,
                 beta: float,
                 cost: float,
                 pi: float,
                 dimension: int,
                 firms: Optional[List[Firm]] = None):
        """
        Initialize Hotelling model on [0,1]^d hypercube.
        
        Args:
            N: Number of firms
            beta: Smoothing parameter for softmin
            cost: Cost parameter for all firms
            pi: Intensity of the Poisson Point Process
            dimension: Dimension of the hypercube
            firms: Optional list of firms (if None, will be initialized)
        """
        self.dimension = dimension
        super().__init__(
            N=N,
            beta=beta,
            cost=cost,
            manifold_volume=1.0,  # [0,1]^d cube has volume 1
            pi=pi,
            firms=firms
        )

    def distance_manifold(self,
                         x: npt.NDArray[np.float64],
                         y: npt.NDArray[np.float64]) -> float:
        """
        Compute L1 distance on [0,1]^d cube.
        For d-dimensional case, this is the sum of absolute differences.
        """
        return np.sum(np.abs(x - y))

    def generate_integration_points(self) -> Tuple[npt.NDArray[np.float64], float]:
        """
        Generate integration points in [0,1]^d cube for numerical integration.
        Uses a grid of points with approximately 10000^(1/d) points per dimension.
        
        Returns:
            Tuple of (points array, weight per point)
        """
        # Calculate points per dimension to maintain roughly similar total points
        points_per_dim = int(np.power(10000, 1/self.dimension))
        
        # Generate grid points for each dimension
        grid_points = [np.linspace(0, 1, points_per_dim) for _ in range(self.dimension)]
        
        # Create meshgrid
        mesh = np.meshgrid(*grid_points)
        
        # Stack coordinates to get points array
        points = np.stack([m.flatten() for m in mesh], axis=1)
        
        # Weight is cube volume divided by number of points
        weight = 1.0 / len(points)
        
        return points, weight

    def get_optimization_bounds(self) -> List[Tuple[float, float]]:
        """
        Get bounds for optimization.
        Position must be in [0,1]^d, price must be above cost.
        """
        position_bounds = [(0, 1) for _ in range(self.dimension)]  # d position bounds
        price_bound = [(self.cost, None)]  # price bound
        return position_bounds + price_bound

    def get_initial_firms(self) -> List[Firm]:
        """
        Get initial positions and prices for firms.
        Positions are randomly distributed in [0,1]^d, prices start at 2*cost.
        """
        firms = []
        for _ in range(self.num_firms):
            # Random position in [0,1]^d
            position = np.random.random(self.dimension)
            # Initial price is twice the cost
            price = 2 * self.cost
            firms.append(Firm(position=position, price=price))
        
        return firms
    
    def get_intial_points(self, firm_index) -> List[Tuple[npt.NDArray[np.float64], float]]:
        return [
            (self.firms[firm_index].position, self.firms[firm_index].price),
            (np.ones(self.dimension), self.firms[firm_index].price)
        ]
    
    def __repr__(self) -> str:
        """String representation of CubeModel structure."""
        header = f"CubeModel(dimension={self.dimension}, beta={self.beta}, cost={self.cost}\n"
        firms_str = "\nFirms:\n"
        for i, firm in enumerate(self.firms):
            firms_str += f"  {i+1}: position={firm.position}, price={firm.price:.3f}\n"
        return header + firms_str + ")"
from qubots.base_problem import BaseProblem
import random
import math
import os

def read_elem(filename):

    # Resolve relative path with respect to this module’s directory.
    if not os.path.isabs(filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_dir, filename)

    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]

def compute_distance_matrix(x_coord, y_coord):
    nb = len(x_coord)
    dist_matrix = [[0 for _ in range(nb)] for _ in range(nb)]
    for i in range(nb):
        for j in range(nb):
            dist_matrix[i][j] = round(math.sqrt((x_coord[i]-x_coord[j])**2 + (y_coord[i]-y_coord[j])**2))
    return dist_matrix

def compute_distance_supplier(x_supplier, y_supplier, x_coord, y_coord):
    nb = len(x_coord)
    dist_supplier = [0 for _ in range(nb)]
    for i in range(nb):
        dist_supplier[i] = round(math.sqrt((x_supplier - x_coord[i])**2 + (y_supplier - y_coord[i])**2))
    return dist_supplier

class InventoryRoutingProblem(BaseProblem):
    """
    Inventory Routing Problem (IRP) for Qubots.
    
    A product must be shipped from a supplier to several customers over a given planning horizon.
    Each customer has a maximum inventory level and a known demand rate. The supplier (following a
    vendor-managed inventory policy) replenishes customers subject to a vehicle capacity constraint.
    Shipments incur transportation costs (based on distances), and both the supplier and customers
    incur inventory holding costs.
    
    **Candidate Solution Representation:**
      A dictionary with keys:
         - "delivery": a 2D list (dimensions: horizon_length × nb_customers) where delivery[t][i] is
           the quantity delivered to customer i at time period t.
         - "route": a list of length horizon_length; each element is a list (a permutation of customer
           indices, 0-indexed) representing the order in which customers are visited at time t.
    """
    
    def __init__(self, instance_file: str):
        (self.nb_customers,
         self.horizon_length,
         self.capacity,
         self.start_level_supplier,
         self.production_rate_supplier,
         self.holding_cost_supplier,
         self.start_level,
         self.max_level,
         self.demand_rate,
         self.holding_cost,
         self.dist_matrix,
         self.dist_supplier) = self._read_instance(instance_file)
    
    def _read_instance(self, instance_file: str):
        """
        Reads the instance file in the Archetti format.
        
        The file format is as follows:
        
        - First line: three integers: 
            * total number of nodes (supplier + customers). 
              (The number of customers is this value minus one.)
            * number of discrete time periods (horizon_length)
            * vehicle capacity.
        
        - Second line: supplier information:
            supplier index, x coordinate, y coordinate, starting inventory level,
            production quantity per time period, unit inventory cost.
        
        - Then, for each customer (in order):
            customer index, x coordinate, y coordinate, starting inventory level,
            maximum inventory level, minimum inventory level, demand rate per time period,
            unit inventory cost.
        
        Returns:
          nb_customers (int), horizon_length (int), capacity (int),
          start_level_supplier (int), production_rate_supplier (int),
          holding_cost_supplier (float), start_level (list of int), max_level (list of int),
          demand_rate (list of int), holding_cost (list of float), 
          dist_matrix (2D list of int), dist_supplier (list of int).
        """
        tokens = read_elem(instance_file)
        it = iter(tokens)
        total_nodes = int(next(it))
        nb_customers = total_nodes - 1
        horizon_length = int(next(it))
        capacity = int(next(it))
        
        # Read supplier info
        supplier_index = int(next(it))  # supplier index (ignored)
        x_coord_supplier = float(next(it))
        y_coord_supplier = float(next(it))
        start_level_supplier = int(next(it))
        production_rate_supplier = int(next(it))
        holding_cost_supplier = float(next(it))
        
        # For each customer, read customer info.
        x_coord = []
        y_coord = []
        start_level = []
        max_level = []
        min_level = []
        demand_rate = []
        holding_cost = []
        for i in range(nb_customers):
            cust_index = int(next(it))  # customer index (ignored)
            x_coord.append(float(next(it)))
            y_coord.append(float(next(it)))
            start_level.append(int(next(it)))
            max_level.append(int(next(it)))
            min_level.append(int(next(it)))  # not used in evaluation
            demand_rate.append(int(next(it)))
            holding_cost.append(float(next(it)))
        
        dist_matrix = compute_distance_matrix(x_coord, y_coord)
        dist_supplier = compute_distance_supplier(x_coord_supplier, y_coord_supplier, x_coord, y_coord)
        return (nb_customers, horizon_length, capacity, start_level_supplier,
                production_rate_supplier, holding_cost_supplier, start_level, max_level,
                demand_rate, holding_cost, dist_matrix, dist_supplier)
    
    def evaluate_solution(self, solution) -> float:
        """
        Evaluates a candidate solution.
        
        The solution must be a dictionary with keys:
          - "delivery": a 2D list of size (horizon_length × nb_customers) with delivery quantities.
          - "route": a list of length horizon_length, where each element is a list (permutation)
                     of customer indices (0-indexed) representing the route at that time period.
        
        The evaluation computes:
          - Inventory levels at the supplier and at each customer over the planning horizon,
            checking that stockout constraints are met.
          - Capacity constraints (total delivered per time period ≤ vehicle capacity).
          - A maximum level constraint for each customer: delivered quantity cannot exceed
            the gap between the customer's max level and its current inventory.
          - Transportation cost computed from the route (distance from supplier to first customer,
            between successive customers, and from the last customer back to the supplier).
          - Total cost as the sum of supplier inventory cost, customer inventory cost, and transportation cost.
        
        If any constraint is violated, a high penalty (1e9) is returned.
        """
        penalty = 1e9
        # Check structure.
        if not isinstance(solution, dict):
            return penalty
        if "delivery" not in solution or "route" not in solution:
            return penalty
        delivery = solution["delivery"]
        route = solution["route"]
        if len(delivery) != self.horizon_length or len(route) != self.horizon_length:
            return penalty
        for t in range(self.horizon_length):
            if len(delivery[t]) != self.nb_customers:
                return penalty
            if not isinstance(route[t], list):
                return penalty
        
        # Compute supplier inventory over time.
        inventory_supplier = [0]*(self.horizon_length+1)
        inventory_supplier[0] = self.start_level_supplier
        for t in range(1, self.horizon_length+1):
            delivered_prev = sum(delivery[t-1])
            inventory_supplier[t] = inventory_supplier[t-1] - delivered_prev + self.production_rate_supplier
            # Constraint: at time t < horizon, supplier inventory must cover next period's delivery.
            if t < self.horizon_length and inventory_supplier[t] < sum(delivery[t]) - 1e-6:
                return penalty
        
        # Compute customer inventories.
        inventory = [[0]*(self.horizon_length+1) for _ in range(self.nb_customers)]
        for i in range(self.nb_customers):
            inventory[i][0] = self.start_level[i]
            for t in range(1, self.horizon_length+1):
                inventory[i][t] = inventory[i][t-1] + delivery[t-1][i] - self.demand_rate[i]
                if inventory[i][t] < -1e-6:
                    return penalty
        
        # Check capacity constraint.
        for t in range(self.horizon_length):
            if sum(delivery[t]) > self.capacity + 1e-6:
                return penalty
        
        # Check maximum level constraints for each customer.
        for t in range(self.horizon_length):
            for i in range(self.nb_customers):
                if delivery[t][i] > self.max_level[i] - inventory[i][t] + 1e-6:
                    return penalty
        
        # Determine which customers receive a delivery at time t.
        # (A customer is considered delivered if it appears in the route for that period.)
        is_delivered = [[False]*self.nb_customers for _ in range(self.horizon_length)]
        for t in range(self.horizon_length):
            for i in route[t]:
                if i < 0 or i >= self.nb_customers:
                    return penalty
                is_delivered[t][i] = True
        
        # Compute transportation cost for each time period.
        dist_routes = [0]*self.horizon_length
        for t in range(self.horizon_length):
            if len(route[t]) > 0:
                seq = route[t]
                d = self.dist_supplier[seq[0]]
                for j in range(1, len(seq)):
                    d += self.dist_matrix[seq[j-1]][seq[j]]
                d += self.dist_supplier[seq[-1]]
                dist_routes[t] = d
            else:
                dist_routes[t] = 0
        
        total_cost_inventory_supplier = self.holding_cost_supplier * sum(inventory_supplier)
        total_cost_inventory = 0
        for i in range(self.nb_customers):
            total_cost_inventory += self.holding_cost[i] * sum(inventory[i])
        total_cost_route = sum(dist_routes)
        objective = total_cost_inventory_supplier + total_cost_inventory + total_cost_route
        return objective
    
    def random_solution(self):
        """
        Generates a random candidate solution.
        
        For each time period, a random delivery is generated (as a vector of random quantities
        between 0 and capacity/nb_customers) and a random route (a random permutation of customer indices).
        Note: This naive generator may produce infeasible solutions.
        """
        delivery = []
        route = []
        for t in range(self.horizon_length):
            delivery_t = [random.uniform(0, self.capacity / self.nb_customers) for _ in range(self.nb_customers)]
            delivery.append(delivery_t)
            perm = list(range(self.nb_customers))
            random.shuffle(perm)
            route.append(perm)
        return {"delivery": delivery, "route": route}

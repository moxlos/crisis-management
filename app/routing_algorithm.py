#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 2023

@author: lefteris

@subject: Optimal Routing Algorithm 

The program

xij​: The amount of emergency help sent from base i to call j.
dij​: The distance between base i and call j.

Supply Constraints: Each base cannot send more help than its available capacity.

∑j xij≤ Capacity of Base i, for i=1,2,...

Demand Constraints: Each call must receive the required amount of help.
∑i xij ≥ 1 of Call j, for j=1,2,…

Non-negativity Constraints:
xij ≥ 0, for i=1,2,… and j=1,2,…

Objective:
    Minimize Si​j ​dij​⋅xij

"""

import os
import pandas as pd
import numpy as np
import pulp
from datetime import datetime

from app.utils import haversine_distance as calculate_distance, SEVERITY_LEVELS


# Linear Programming formulation for documentation
LP_FORMULATION = """
LINEAR PROGRAMMING FORMULATION
==============================

Decision Variables:
    x_ij = Amount of emergency help sent from base i to emergency j
           (Integer, x_ij >= 0)

Parameters:
    d_ij = Distance between base i and emergency j (km)
    C_i  = Capacity of base i (maximum units available)
    n    = Number of bases
    m    = Number of emergencies

Objective Function:
    Minimize Z = Σ_i Σ_j (d_ij × x_ij)

    Minimize total distance traveled for all assignments.

Subject to:

1. Supply Constraints (each base cannot exceed capacity):
    Σ_j x_ij <= C_i    for all i = 1, 2, ..., n

2. Demand Constraints (each emergency must receive help):
    Σ_i x_ij >= 1      for all j = 1, 2, ..., m

3. Non-negativity Constraints:
    x_ij >= 0          for all i, j
    x_ij ∈ Z           (integer values)

Mathematical Notation:
    min  Σᵢ₌₁ⁿ Σⱼ₌₁ᵐ dᵢⱼ·xᵢⱼ
    s.t. Σⱼ₌₁ᵐ xᵢⱼ ≤ Cᵢ     ∀i ∈ {1,...,n}
         Σᵢ₌₁ⁿ xᵢⱼ ≥ 1      ∀j ∈ {1,...,m}
         xᵢⱼ ≥ 0, xᵢⱼ ∈ ℤ   ∀i,j
"""



class OptimalEm:
    def __init__(self, bases_df, em_df):
        #All arrays
        self.bases_coord = np.array(bases_df[['coordinates_lat',
                                              'coordinates_long']])
        self.em_coord = np.array(em_df[['coordinates_lat',
                                              'coordinates_long']])
        self.nVar = bases_df.shape[0]
        self.mVar = em_df.shape[0]
        self.base_capacity = np.array(bases_df['capacity'])
        self.base_names = np.array(bases_df['name'])
        self.bases_df = bases_df
        self.em_df = em_df
        self.dist_matrix = np.zeros((self.nVar, self.mVar))
        # Triage tracking
        self.queued_emergencies = pd.DataFrame()
        self.triage_applied = False
        
    def get_variables(self):
        """Create decision variables using array indices (not database IDs)."""
        xshape = (range(self.nVar), range(self.mVar))
        x = pulp.LpVariable.dicts("X", xshape, lowBound=0, cat="Integer")
        return x
    
    def calc_distances(self):
        #TODO: make more efficient numba?
        
        N,M = self.dist_matrix.shape
        for i in range(N):
            for j in range(M):
              self.dist_matrix[i,j] = calculate_distance(self.bases_coord[i,0],
                                                    self.bases_coord[i,1],
                                                    self.em_coord[j,0],
                                                    self.em_coord[j,1])
    
    
    def get_solution(self):
        """
        Solve the linear programming optimization problem.

        Returns:
            pulp.LpProblem: Solved optimization problem
        """
        # Calculate distances (modifies self.dist_matrix)
        self.calc_distances()

        x = self.get_variables()

        prob = pulp.LpProblem("distribution_opt", pulp.LpMinimize)

        # Objective: Minimize total distance
        objective_function = pulp.lpSum([
            self.dist_matrix[n_idx, m_idx] * x[n_idx][m_idx]
            for n_idx in range(self.nVar)
            for m_idx in range(self.mVar)
        ])

        prob += objective_function

        # Constraints
        # Supply constraint: Each base cannot exceed its capacity
        for n_idx in range(self.nVar):
            prob += sum(x[n_idx][m_idx] for m_idx in range(self.mVar)) <= self.base_capacity[n_idx]

        # Demand constraint: Each call must receive at least 1 unit of help
        for m_idx in range(self.mVar):
            prob += sum(x[n_idx][m_idx] for n_idx in range(self.nVar)) >= 1

        prob.solve(pulp.apis.PULP_CBC_CMD(msg=False))

        # Store solver status for later reference
        self.solver_status = pulp.LpStatus[prob.status]
        self.is_optimal = prob.status == pulp.LpStatusOptimal

        # Check solver status
        if prob.status != pulp.LpStatusOptimal:
            print(f"Warning: LP solver status = {pulp.LpStatus[prob.status]}")
            # Check if problem is infeasible (more emergencies than total capacity)
            total_capacity = sum(self.base_capacity)
            if total_capacity < self.mVar:
                print(f"Problem is likely infeasible: total capacity ({total_capacity}) < emergencies ({self.mVar})")

        return prob
    
    def get_network(self):
        """
        Extract network assignments from solved optimization.

        Returns:
            tuple: (prob, Xfilt) - solved problem and filtered assignments DataFrame
        """
        prob = self.get_solution()
        self.prob = prob  # Store for later access by logging methods

        X = {'base_id': [],
             'em_id': [],
             'value': []}

        # Map array indices back to database IDs
        base_ids = list(self.bases_df['id'])
        em_ids = list(self.em_df['id'])

        for v in prob.variables():
            if v.name.startswith('X_'):
                # Extract indices from variable name (e.g., "X_0_5" -> [0, 5])
                parts = v.name[2:].split("_")
                base_idx = int(parts[0])
                em_idx = int(parts[1])

                # Map indices to actual database IDs
                X['base_id'].append(base_ids[base_idx])
                X['em_id'].append(em_ids[em_idx])
                X['value'].append(v.varValue)

        X = pd.DataFrame(X)
        Xfilt = X.loc[X['value'] > 0].reset_index(drop=True)
        return (prob, Xfilt)
    
    
    def get_report(self, write_log=True):
        """
        Generate assignment report with distances.

        Args:
            write_log: If True, write LP formulation and results to log files

        Returns:
            pd.DataFrame: Report with base_id, em_id, supply, distance, and nearest_distance
        """
        prob, X = self.get_network()

        X = X.rename(columns={'value': 'supply'})

        # Write logs if requested
        if write_log:
            try:
                self.write_logs(prob, X)
            except IOError as e:
                print(f"Warning: Failed to write optimization logs: {e}")

        # Validate capacity constraints
        base_ids = list(self.bases_df['id'])
        base_id_to_capacity = dict(zip(self.bases_df['id'], self.base_capacity))

        # Check if any base exceeds capacity
        base_totals = X.groupby('base_id')['supply'].sum()
        for base_id, total_supply in base_totals.items():
            capacity = base_id_to_capacity.get(base_id, 0)
            if total_supply > capacity:
                print(f"WARNING: Base {base_id} assigned {total_supply} units but has capacity {capacity}")
                print(f"  Solver status: {self.solver_status}")

        # Map indices back to database IDs
        base_ids = list(self.bases_df['id'])
        em_ids = list(self.em_df['id'])

        # Look up distance directly from the matrix using the indices
        base_id_to_idx = {bid: idx for idx, bid in enumerate(base_ids)}
        em_id_to_idx = {eid: idx for idx, eid in enumerate(em_ids)}

        distances = []
        nearest_distances = []
        nearest_base_ids = []

        for _, row in X.iterrows():
            base_idx = base_id_to_idx[row['base_id']]
            em_idx = em_id_to_idx[row['em_id']]
            distances.append(self.dist_matrix[base_idx, em_idx])

            # Find the nearest base for this emergency (for comparison)
            em_distances = self.dist_matrix[:, em_idx]
            nearest_idx = np.argmin(em_distances)
            nearest_distances.append(em_distances[nearest_idx])
            nearest_base_ids.append(base_ids[nearest_idx])

        X['distance'] = distances
        X['nearest_distance'] = nearest_distances
        X['nearest_base_id'] = nearest_base_ids

        return X

    def get_report_with_triage(self, write_log=True):
        """
        Generate assignment report with triage support.

        When total capacity < number of emergencies, applies triage:
        1. Sort emergencies by severity (1=Critical first, then 2=Urgent, then 3=Normal)
        2. Select highest priority emergencies up to total capacity
        3. Run optimization on selected subset
        4. Queue remaining emergencies

        Args:
            write_log: If True, write LP formulation and results to log files

        Returns:
            dict: {
                'assignments': DataFrame with assignments,
                'queued': DataFrame with queued emergencies,
                'triage_applied': bool,
                'solver_status': str,
                'is_optimal': bool
            }
        """
        total_capacity = int(sum(self.base_capacity))
        num_emergencies = self.mVar

        # Check if triage is needed
        if total_capacity >= num_emergencies:
            # No triage needed - run normal optimization
            report = self.get_report(write_log=write_log)
            return {
                'assignments': report,
                'queued': pd.DataFrame(),
                'triage_applied': False,
                'solver_status': self.solver_status,
                'is_optimal': self.is_optimal,
                'total_capacity': total_capacity,
                'num_emergencies': num_emergencies
            }

        # Triage needed - sort by severity and select top priorities
        self.triage_applied = True

        # Ensure severity column exists, default to 2 (Urgent) if missing
        if 'severity' not in self.em_df.columns:
            self.em_df['severity'] = 2

        # Sort by severity (1=Critical first), then by datetime (oldest first)
        em_sorted = self.em_df.sort_values(
            by=['severity', 'dt'],
            ascending=[True, True]
        ).reset_index(drop=True)

        # Split into assigned (up to capacity) and queued (rest)
        em_to_assign = em_sorted.head(total_capacity).copy()
        em_queued = em_sorted.tail(num_emergencies - total_capacity).copy()

        self.queued_emergencies = em_queued

        # Create new optimizer with reduced emergency set
        triage_opt = OptimalEm(self.bases_df, em_to_assign)
        triage_opt.triage_applied = True
        triage_opt.queued_emergencies = em_queued

        # Run optimization on subset
        report = triage_opt.get_report(write_log=False)

        # Write logs with triage info
        if write_log:
            try:
                triage_opt.write_logs_with_triage(
                    report, em_queued, num_emergencies, total_capacity
                )
            except IOError as e:
                print(f"Warning: Failed to write optimization logs: {e}")

        # Add severity to queued emergencies output
        em_queued_out = em_queued[['id', 'name', 'phone_number', 'message',
                                    'coordinates_lat', 'coordinates_long',
                                    'severity', 'dt']].copy()
        em_queued_out['severity_label'] = em_queued_out['severity'].map(SEVERITY_LEVELS)

        return {
            'assignments': report,
            'queued': em_queued_out,
            'triage_applied': True,
            'solver_status': triage_opt.solver_status,
            'is_optimal': triage_opt.is_optimal,
            'total_capacity': total_capacity,
            'num_emergencies': num_emergencies,
            'assigned_count': len(em_to_assign),
            'queued_count': len(em_queued)
        }

    def write_logs_with_triage(self, X, queued_df, total_emergencies, total_capacity):
        """Write optimization results with triage information.

        Args:
            X: DataFrame with assignments
            queued_df: DataFrame with queued emergencies
            total_emergencies: Total number of emergencies before triage
            total_capacity: Total base capacity
        """
        log_dir = "data/opt_log"

        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            raise IOError(f"Failed to create log directory '{log_dir}': {e}")

        # Write LP problem formulation file
        lp_file = os.path.join(log_dir, "pulp_problem.lp")
        try:
            if hasattr(self, 'prob') and self.prob is not None:
                self.prob.writeLP(lp_file)
        except Exception as e:
            print(f"Warning: Failed to write LP file: {e}")

        results_file = os.path.join(log_dir, "optimization_results.txt")
        try:
            with open(results_file, "w") as file:
                file.write("=" * 60 + "\n")
                file.write("EMERGENCY ROUTING OPTIMIZATION RESULTS (WITH TRIAGE)\n")
                file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                file.write("=" * 60 + "\n\n")

                # Write mathematical formulation
                file.write(LP_FORMULATION)
                file.write("\n\n")

                # Triage summary
                file.write("TRIAGE SUMMARY\n")
                file.write("-" * 40 + "\n")
                file.write(f"Total Emergencies:      {total_emergencies}\n")
                file.write(f"Total Base Capacity:    {total_capacity}\n")
                file.write(f"Capacity Shortage:      {total_emergencies - total_capacity}\n")
                file.write(f"Emergencies Assigned:   {self.mVar}\n")
                file.write(f"Emergencies Queued:     {len(queued_df)}\n\n")

                # Problem summary
                file.write("OPTIMIZATION PROBLEM (after triage)\n")
                file.write("-" * 40 + "\n")
                file.write(f"Number of Bases (n):        {self.nVar}\n")
                file.write(f"Number of Emergencies (m):  {self.mVar}\n")
                file.write(f"Decision Variables:         {self.nVar * self.mVar} (X_i_j)\n")
                file.write(f"Supply Constraints:         {self.nVar}\n")
                file.write(f"Demand Constraints:         {self.mVar}\n\n")

                # Capacity info
                file.write("BASE CAPACITIES\n")
                file.write("-" * 40 + "\n")
                base_ids = list(self.bases_df['id'])
                for i, (base_id, cap) in enumerate(zip(base_ids, self.base_capacity)):
                    file.write(f"Base {base_id} ({self.base_names[i]}): {cap}\n")
                file.write(f"\nTotal Capacity: {total_capacity}\n\n")

                # Solver results
                file.write("SOLVER RESULTS\n")
                file.write("-" * 40 + "\n")
                file.write(f"Status:            {self.solver_status}\n")
                file.write(f"Is Optimal:        {self.is_optimal}\n\n")

                # Assignment summary
                file.write("ASSIGNMENTS SUMMARY\n")
                file.write("-" * 40 + "\n")
                if len(X) > 0:
                    base_totals = X.groupby('base_id')['supply'].sum()
                    for base_id in base_ids:
                        assigned = base_totals.get(base_id, 0)
                        cap = self.base_capacity[base_ids.index(base_id)]
                        status = "OK" if assigned <= cap else "VIOLATION!"
                        file.write(f"Base {base_id}: {int(assigned)}/{int(cap)} units [{status}]\n")
                else:
                    file.write("No assignments made.\n")

                # Queued emergencies
                file.write("\nQUEUED EMERGENCIES (by severity)\n")
                file.write("-" * 40 + "\n")
                file.write(f"{'ID':<6} {'Severity':<12} {'Name':<20} {'Phone':<15}\n")
                file.write("-" * 40 + "\n")
                for _, row in queued_df.iterrows():
                    sev_label = SEVERITY_LEVELS.get(row.get('severity', 2), 'Unknown')
                    file.write(f"{row['id']:<6} {sev_label:<12} {str(row['name'])[:20]:<20} {str(row['phone_number'])[:15]:<15}\n")

                file.write("\n" + "=" * 60 + "\n")
                file.write(f"LP formulation written to: {lp_file}\n")
                file.write("=" * 60 + "\n")

        except IOError as e:
            raise IOError(f"Failed to write results to '{results_file}': {e}")

    def write_failure_log(self, reason, details=None):
        """Write log for failed optimization attempts.

        Args:
            reason: Short description of failure reason
            details: Additional details dict
        """
        log_dir = "data/opt_log"

        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError:
            return  # Silently fail if can't create directory

        results_file = os.path.join(log_dir, "optimization_results.txt")
        try:
            with open(results_file, "w") as file:
                file.write("=" * 60 + "\n")
                file.write("EMERGENCY ROUTING OPTIMIZATION - FAILED\n")
                file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                file.write("=" * 60 + "\n\n")

                file.write("FAILURE REASON\n")
                file.write("-" * 40 + "\n")
                file.write(f"{reason}\n\n")

                if details:
                    file.write("DETAILS\n")
                    file.write("-" * 40 + "\n")
                    for key, value in details.items():
                        file.write(f"{key}: {value}\n")
                    file.write("\n")

                # Problem info
                file.write("PROBLEM INFO\n")
                file.write("-" * 40 + "\n")
                file.write(f"Number of Bases:        {self.nVar}\n")
                file.write(f"Number of Emergencies:  {self.mVar}\n")
                total_cap = int(sum(self.base_capacity))
                file.write(f"Total Capacity:         {total_cap}\n")
                file.write(f"Capacity Shortage:      {self.mVar - total_cap}\n\n")

                # Base details
                file.write("BASE CAPACITIES\n")
                file.write("-" * 40 + "\n")
                base_ids = list(self.bases_df['id'])
                for i, (base_id, cap) in enumerate(zip(base_ids, self.base_capacity)):
                    file.write(f"Base {base_id} ({self.base_names[i]}): {cap}\n")

                file.write("\n" + "=" * 60 + "\n")
        except IOError:
            pass  # Silently fail

    def write_logs(self, prob, X):
        """Write optimization results and problem formulation to log files.

        Creates two files in data/opt_log/:
        - pulp_problem.lp: Linear programming formulation
        - optimization_results.txt: Solution summary with status and optimal values

        Args:
            prob: PuLP problem object with solved optimization
            X: DataFrame with assignments (base_id, em_id, supply)

        Raises:
            IOError: If log directory cannot be created or files cannot be written
        """
        log_dir = "data/opt_log"

        # Create log directory if it doesn't exist
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            raise IOError(f"Failed to create log directory '{log_dir}': {e}")

        # Write LP problem formulation
        lp_file = os.path.join(log_dir, "pulp_problem.lp")
        try:
            prob.writeLP(lp_file)
        except Exception as e:
            raise IOError(f"Failed to write LP file to '{lp_file}': {e}")

        # Write optimization results
        results_file = os.path.join(log_dir, "optimization_results.txt")
        try:
            with open(results_file, "w") as file:
                file.write("=" * 60 + "\n")
                file.write("EMERGENCY ROUTING OPTIMIZATION RESULTS\n")
                file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                file.write("=" * 60 + "\n\n")

                # Write mathematical formulation
                file.write(LP_FORMULATION)
                file.write("\n\n")

                # Problem summary
                file.write("PROBLEM INSTANCE\n")
                file.write("-" * 40 + "\n")
                file.write(f"Number of Bases (n):        {self.nVar}\n")
                file.write(f"Number of Emergencies (m):  {self.mVar}\n")
                file.write(f"Decision Variables:         {self.nVar * self.mVar} (X_i_j)\n")
                file.write(f"Supply Constraints:         {self.nVar}\n")
                file.write(f"Demand Constraints:         {self.mVar}\n")
                file.write(f"Total Constraints:          {self.nVar + self.mVar}\n\n")

                # Capacity info
                file.write("BASE CAPACITIES\n")
                file.write("-" * 40 + "\n")
                base_ids = list(self.bases_df['id'])
                total_capacity = 0
                for i, (base_id, cap) in enumerate(zip(base_ids, self.base_capacity)):
                    file.write(f"Base {base_id} ({self.base_names[i]}): {cap}\n")
                    total_capacity += cap
                file.write(f"\nTotal Capacity: {total_capacity}\n")
                file.write(f"Total Demand:   {self.mVar} (1 unit per emergency)\n")
                if total_capacity >= self.mVar:
                    file.write(f"Feasibility:    OK (capacity >= demand)\n\n")
                else:
                    file.write(f"Feasibility:    INFEASIBLE (capacity < demand by {self.mVar - total_capacity})\n\n")

                # Solver results
                file.write("SOLVER RESULTS\n")
                file.write("-" * 40 + "\n")
                file.write(f"Status:                 {pulp.LpStatus[prob.status]}\n")
                file.write(f"Optimal Objective:      {pulp.value(prob.objective):.4f} km (total distance)\n\n")

                # Assignment summary
                file.write("ASSIGNMENTS SUMMARY\n")
                file.write("-" * 40 + "\n")
                base_totals = X.groupby('base_id')['supply'].sum()
                for base_id in base_ids:
                    assigned = base_totals.get(base_id, 0)
                    cap = self.base_capacity[base_ids.index(base_id)]
                    status = "OK" if assigned <= cap else "VIOLATION!"
                    file.write(f"Base {base_id}: {int(assigned)}/{int(cap)} units assigned [{status}]\n")

                # All non-zero variable values
                file.write("\nNON-ZERO DECISION VARIABLES\n")
                file.write("-" * 40 + "\n")
                file.write(f"{'Variable':<15} {'Value':<10} {'Distance (km)':<15}\n")
                file.write("-" * 40 + "\n")

                # Map for looking up distances
                base_id_to_idx = {bid: idx for idx, bid in enumerate(base_ids)}
                em_ids = list(self.em_df['id'])
                em_id_to_idx = {eid: idx for idx, eid in enumerate(em_ids)}

                for _, row in X.iterrows():
                    base_idx = base_id_to_idx[row['base_id']]
                    em_idx = em_id_to_idx[row['em_id']]
                    dist = self.dist_matrix[base_idx, em_idx]
                    var_name = f"X_{row['base_id']}_{row['em_id']}"
                    file.write(f"{var_name:<15} {int(row['supply']):<10} {dist:<15.4f}\n")

                file.write("\n" + "=" * 60 + "\n")
                file.write(f"LP formulation written to: {lp_file}\n")
                file.write("=" * 60 + "\n")

        except IOError as e:
            raise IOError(f"Failed to write results to '{results_file}': {e}")

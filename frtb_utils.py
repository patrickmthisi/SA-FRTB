# Author: Patrick L. Mthisi
# Contact details: patrickmthisi@hotmail.com

import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm


# FRTB CALCULATION ENGINE 
class FRTBCalculator:
    # Define detailed risk weights (Basel Committee on Banking Supervision)
    RISK_WEIGHTS = {
        'IR': {
            '0.25y': 1.21, '1y': 1.43, '3y': 1.76, '5y': 2.08, 
            '10y': 2.55, '20y': 2.77, '30y': 2.86
        },
        'FX': 7.5,  # Same for all FX pairs
        'EQ': {
            'Technology': 33.0, 'Financial': 30.0, 'Energy': 34.0,
            'Healthcare': 32.0, 'Consumer': 31.0, 'Industrial': 30.5, 'Utilities': 29.0
        },
        'CO': {
            'Energy': 20.0, 'Metals': 18.0, 'Agriculture': 22.0, 'Livestock': 25.0
        },
        'CR': {
            'AAA': 0.5, 'AA': 1.0, 'A': 1.5, 'BBB': 3.0,
            'BB': 7.0, 'B': 12.0, 'CCC': 20.0, 'CC': 35.0, 'C': 50.0, 'D': 100.0
        }
    }
    
    # Vega risk weights (typically same as delta risk weights)
    VEGA_RISK_WEIGHTS = {
        'IR': 0.21,  
        'FX': 0.35,
        'EQ': 0.55,
        'CO': 0.40,
        'CR': 0.28
    }
    
    # Liquidity horizons (days)
    LH = {
        'IR': 20, 'FX': 10, 'EQ': 20, 'CO': 20, 'CR': 60
    }
    
    # Curvature risk parameters
    CURVATURE_STRESS = {
        'IR': 0.65, 'FX': 0.30, 'EQ': 0.55, 'CO': 0.60, 'CR': 0.85
    }
    
    # Correlations within buckets
    CORRELATIONS = {
        'IR': 0.5,   # Higher correlation within same currency
        'FX': 0.3,   # Lower correlation between currency pairs
        'EQ': 0.4,   # Moderate correlation within sectors
        'CO': 0.6,   # Higher correlation within commodity types
        'CR': 0.7    # High correlation within credit ratings
    }
    
    # Vega correlations 
    VEGA_CORRELATIONS = {
        'IR': 0.3, 'FX': 0.2, 'EQ': 0.25, 'CO': 0.35, 'CR': 0.4
    }
    
    # Correlations across buckets 
    CROSS_BUCKET_CORR = {
        ('IR', 'FX'): 0.3, ('IR', 'EQ'): 0.1, ('IR', 'CO'): 0.2, ('IR', 'CR'): 0.4,
        ('FX', 'EQ'): 0.2, ('FX', 'CO'): 0.3, ('FX', 'CR'): 0.1,
        ('EQ', 'CO'): 0.4, ('EQ', 'CR'): 0.3,
        ('CO', 'CR'): 0.2
    }
    
    # Vega cross-bucket correlations 
    VEGA_CROSS_BUCKET_CORR = {
        ('IR', 'FX'): 0.15, ('IR', 'EQ'): 0.05, ('IR', 'CO'): 0.1, ('IR', 'CR'): 0.2,
        ('FX', 'EQ'): 0.1, ('FX', 'CO'): 0.15, ('FX', 'CR'): 0.05,
        ('EQ', 'CO'): 0.2, ('EQ', 'CR'): 0.15,
        ('CO', 'CR'): 0.1
    }
    
    def __init__(self, base_currency='USD'):
        self.base_currency = base_currency
    
    def get_risk_weight(self, asset_class, bucket):
        """Get risk weight considering liquidity horizon adjustment"""
        if asset_class == 'FX':
            rw = self.RISK_WEIGHTS['FX']
        elif asset_class == 'IR':
            rw = self.RISK_WEIGHTS['IR'][bucket.split('_')[1]]
        elif asset_class == 'EQ':
            rw = self.RISK_WEIGHTS['EQ'][bucket.split('_')[1]]
        elif asset_class == 'CO':
            rw = self.RISK_WEIGHTS['CO'][bucket.split('_')[1]]
        else:  # CR
            rw = self.RISK_WEIGHTS['CR'][bucket.split('_')[1]]
        
        # Adjust for liquidity horizon
        lh_ratio = np.sqrt(self.LH[asset_class] / 10)
        return rw * lh_ratio
    
    def get_vega_risk_weight(self, asset_class):
        """Get vega risk weight with liquidity horizon adjustment"""
        vrw = self.VEGA_RISK_WEIGHTS[asset_class]
        lh_ratio = np.sqrt(self.LH[asset_class] / 10)
        return vrw * lh_ratio
    
    def calculate_delta_risk(self, book):
        """Calculate delta risk with granular buckets and cross-bucket correlations"""
        # Filter only non-zero sensitivities
        delta_book = book[book['Sensitivity'] != 0]
        
        # Step 1: Calculate within-bucket risk
        bucket_risk = {}
        bucket_sensitivities = {}
        
        for (asset_class, bucket), group in delta_book.groupby(['AssetClass', 'Bucket']):
            RW = self.get_risk_weight(asset_class, bucket)
            S_k = group['Sensitivity'].sum()
            SS_k = (group['Sensitivity']**2).sum()
            rho = self.CORRELATIONS[asset_class]
            
            # Within-bucket capital
            K_b = np.sqrt(SS_k + rho * S_k**2) * RW
            bucket_risk[(asset_class, bucket)] = K_b
            bucket_sensitivities[(asset_class, bucket)] = S_k
        
        # Step 2: Calculate across-bucket risk
        buckets = list(bucket_risk.keys())
        n = len(buckets)
        if n == 0:
            return {'total': 0}
        
        corr_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i+1, n):
                asset_i = buckets[i][0]
                asset_j = buckets[j][0]
                corr = self.CROSS_BUCKET_CORR.get((asset_i, asset_j), 0)
                if (asset_j, asset_i) in self.CROSS_BUCKET_CORR:
                    corr = self.CROSS_BUCKET_CORR[(asset_j, asset_i)]
                corr_matrix[i, j] = corr_matrix[j, i] = corr
        
        # Calculate cross-bucket risk
        K = np.array([bucket_risk[b] for b in buckets])
        S = np.array([bucket_sensitivities[b] for b in buckets])
        cross_risk = np.sqrt(S @ corr_matrix @ S)
        
        total_delta_risk = np.sqrt(np.sum(K**2) + cross_risk)
        
        return {
            'within_bucket': bucket_risk,
            'cross_bucket': cross_risk,
            'total': total_delta_risk
        }
    
    def calculate_vega_risk(self, book):
        """Calculate vega risk for options positions"""
        # Filter only positions with vega sensitivity
        vega_book = book[book['Vega'] != 0]
        if vega_book.empty:
            return {'total': 0}
        
        # Step 1: Calculate within-bucket risk
        bucket_risk = {}
        bucket_vegas = {}
        
        for (asset_class, bucket), group in vega_book.groupby(['AssetClass', 'Bucket']):
            RW = self.get_vega_risk_weight(asset_class)
            V_k = group['Vega'].sum()
            VV_k = (group['Vega']**2).sum()
            rho = self.VEGA_CORRELATIONS[asset_class]
            
            # Within-bucket capital
            K_b = np.sqrt(VV_k + rho * V_k**2) * RW
            bucket_risk[(asset_class, bucket)] = K_b
            bucket_vegas[(asset_class, bucket)] = V_k
        
        # Step 2: Calculate across-bucket risk
        buckets = list(bucket_risk.keys())
        n = len(buckets)
        corr_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i+1, n):
                asset_i = buckets[i][0]
                asset_j = buckets[j][0]
                corr = self.VEGA_CROSS_BUCKET_CORR.get((asset_i, asset_j), 0)
                if (asset_j, asset_i) in self.VEGA_CROSS_BUCKET_CORR:
                    corr = self.VEGA_CROSS_BUCKET_CORR[(asset_j, asset_i)]
                corr_matrix[i, j] = corr_matrix[j, i] = corr
        
        # Calculate cross-bucket risk
        K = np.array([bucket_risk[b] for b in buckets])
        V = np.array([bucket_vegas[b] for b in buckets])
        cross_risk = np.sqrt(V @ corr_matrix @ V)
        
        total_vega_risk = np.sqrt(np.sum(K**2) + cross_risk)
        
        return {
            'within_bucket': bucket_risk,
            'cross_bucket': cross_risk,
            'total': total_vega_risk
        }
    
    def calculate_curvature_risk(self, book):
        """Calculate curvature risk using stress scenarios"""
        curvature_risk = 0
        curvature_results = []
        
        for idx, position in book.iterrows():
            asset_class = position['AssetClass']
            S = position['Sensitivity']
            RW = self.get_risk_weight(asset_class, position['Bucket'])
            stress = self.CURVATURE_STRESS[asset_class]
            
            # Calculate curvature add-on
            CVR = max(0, S * stress * RW - S * RW)
            
            # Apply scaling factor
            curvature_risk += CVR
            curvature_results.append({
                'PositionID': position['PositionID'],
                'CurvatureRisk': CVR
            })
        
        return {
            'total': curvature_risk,
            'position_details': pd.DataFrame(curvature_results)
        }
    
    def calculate_default_risk(self, book):
        """Calculate jump-to-default risk for credit exposures"""
        credit_book = book[book['AssetClass'] == 'CR'].copy()
        if credit_book.empty:
            return 0
        
        # Calculate JTD = max(0, LGD × notional) for long positions
        # For short positions: JTD = min(0, LGD × notional)
        credit_book['JTD'] = credit_book.apply(
            lambda x: x['LGD'] * x['Notional'] if x['Sensitivity'] >= 0 
            else -x['LGD'] * x['Notional'], axis=1
        )
        
        # Aggregate by credit rating bucket
        default_risk = 0
        for rating, group in credit_book.groupby('Bucket'):
            rw = self.RISK_WEIGHTS['CR'][rating.split('_')[1]]
            JTD_sum = group['JTD'].sum()
            
            # Calculate capital requirement
            K = 1.06 * rw * abs(JTD_sum) / 10000  # Scaling to appropriate magnitude
            default_risk += K
        
        return default_risk
    
    def calculate_total_capital(self, book):
        """Calculate total capital requirement with all components"""
        # Convert to base currency (simplified)
        # In practice, we'd use FX rates and convert all positions
        fx_converted_book = book.copy()
        
        # Calculate components
        delta_results = self.calculate_delta_risk(fx_converted_book)
        vega_results = self.calculate_vega_risk(fx_converted_book)
        curvature_results = self.calculate_curvature_risk(fx_converted_book)
        default_risk = self.calculate_default_risk(fx_converted_book)
        
        total_capital = (
            delta_results['total'] + 
            vega_results['total'] + 
            curvature_results['total'] + 
            default_risk
        )
        
        return {
            'DeltaRisk': delta_results,
            'VegaRisk': vega_results,
            'CurvatureRisk': curvature_results,
            'DefaultRisk': default_risk,
            'TotalCapital': total_capital
        }
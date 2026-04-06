import numpy as np

def calculate_modified_risk(base_r, alpha_list, adjacency_w):
    """
    論文に基づくリスク計算エンジン
    m = (I - AW)^-1 (I - A)r
    
    Args:
        base_r (np.ndarray): 固有リスクベクトル (n x 1)
        alpha_list (np.ndarray): 感受性ベクトル (n x 1)
        adjacency_w (np.ndarray): 依存度行列 (n x n, W_ij = iがjに依存する割合)
        
    Returns:
        np.ndarray: 変容したリスクスコアベクトル m (n x 1)
    """
    n = len(base_r)
    I = np.eye(n)
    
    # A = diag(alpha)
    A = np.diag(alpha_list)
    
    # (I - AW)
    matrix_to_invert = I - np.dot(A, adjacency_w)
    
    try:
        # 逆行列の計算
        inv_matrix = np.linalg.inv(matrix_to_invert)
        
        # m = (I - AW)^-1 * r
        m = np.dot(inv_matrix, base_r)
        
        # リスクが負にならないようにクリップ (数値誤差対策)
        m = np.maximum(m, 0)
        
        return m
    except np.linalg.LinAlgError:
        # 逆行列が存在しない（爆発的リスク）場合のフォールバック
        return base_r * 10.0 # 暫定的なエラー表示としての高いリスク値

def get_contagion_ratio(m, r):
    """リスク増幅率 (m/r) の計算"""
    # rが極小の場合のゼロ除算回避
    safe_r = np.where(r < 0.00001, 0.00001, r)
    return m / safe_r

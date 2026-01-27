import torch

def sim3_umeyama(pred, gt):
    """
    pred, gt: (F, 3) points
    Returns scale s, rotation R (3x3), translation t (3,)
    """
    mu_pred = pred.mean(dim=0, keepdim=True)
    mu_gt   = gt.mean(dim=0, keepdim=True)

    Xc = pred - mu_pred
    Yc = gt - mu_gt

    cov = (Yc.T @ Xc) / pred.shape[0]
    U, D, Vt = torch.linalg.svd(cov)
    S = torch.eye(3, device=pred.device)
    if torch.linalg.det(U) * torch.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = (D * torch.diag(S)).sum() / (Xc**2).sum()
    t = mu_gt.squeeze() - scale * (R @ mu_pred.squeeze())
    return scale, R, t

def align_sim3(pred_C2W, gt_C2W):
    pred_t = pred_C2W[:, :3, 3]
    gt_t   = gt_C2W[:, :3, 3]
    s, R, t = sim3_umeyama(pred_t, gt_t)

    pred_aligned = pred_C2W.clone()
    pred_aligned[:, :3, :3] = R @ pred_aligned[:, :3, :3]
    pred_aligned[:, :3, 3] = (s * (R @ pred_C2W[:, :3, 3].T)).T + t
    return pred_aligned

def compute_ATE(pred_C2W, gt_C2W):
    pred_aligned = align_sim3(pred_C2W, gt_C2W)
    ate = torch.norm(gt_C2W[:, :3, 3] - pred_aligned[:, :3, 3], dim=1)
    return torch.sqrt((ate**2).mean()), pred_aligned

def compute_RPE_RRE(pred_C2W, gt_C2W, delta=1):
    F = pred_C2W.shape[0]
    
    # Relative transformations
    T_gt_rel   = torch.linalg.inv(gt_C2W[:-delta]) @ gt_C2W[delta:]
    T_pred_rel = torch.linalg.inv(pred_C2W[:-delta]) @ pred_C2W[delta:]

    # Error matrix
    E = torch.linalg.inv(T_gt_rel) @ T_pred_rel  # (F-delta, 4, 4)

    # Translation error
    rpe_trans = torch.norm(E[:, :3, 3], dim=1)
    RPE_trans_mean = rpe_trans.mean()

    # Rotation error (SO3 geodesic)
    R_err = E[:, :3, :3]
    cos_theta = (R_err[:,0,0] + R_err[:,1,1] + R_err[:,2,2] - 1)/2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    rpe_rot = torch.acos(cos_theta)
    RRE_rot_mean = rpe_rot.mean()

    return RPE_trans_mean, RRE_rot_mean

# # pred_C2W, gt_C2W: (F, 4, 4) PyTorch Tensor
# ATE, pred_aligned = compute_ATE(pred_C2W, gt_C2W)
# RPE_trans, RRE_rot = compute_RPE_RRE(pred_aligned, gt_C2W, delta=1)

# print("ATE RMSE:", ATE.item())
# print("RPE translation mean:", RPE_trans.item())
# print("RRE rotation mean (deg):", torch.rad2deg(RRE_rot).item())

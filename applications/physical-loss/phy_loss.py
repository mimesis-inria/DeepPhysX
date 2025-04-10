import torch
from einops import rearrange


class PhysicalLoss(torch.nn.Module):

    def __init__(self, mu, lmbd, c1 = 0.5, c2 = 0.5):

        torch.nn.Module.__init__(self)

        self.mse_loss = torch.nn.MSELoss()
        self.phy_loss = divPK1NeoHookLoss(mu=mu, lmbd=lmbd)
        self.c1, self.c2 = c1, c2

    def init_loss_coeff(self, prediction: torch.Tensor, target: torch.Tensor) -> None:

        # Compute each loss function for N samples
        mse_values = []
        phy_values = []
        for p, t in zip(prediction, target):
            p = p.reshape((1, *p.shape))
            t = t.reshape((1, *t.shape))
            mse_values.append(self.mse_loss.forward(p, t))
            phy_values.append(self.phy_loss.forward(p))
        loss_values = torch.tensor([mse_values, phy_values])

        # Scale coeffs to match the desired ratio
        original_coeff = torch.tensor([self.c1, self.c2])
        scaled_coeff = original_coeff / torch.mean(loss_values, dim=1)
        # Normalize coeffs
        scaled_coeff *= (torch.sum(original_coeff) / torch.sum(scaled_coeff))

        self.c1, self.c2 = scaled_coeff

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        mse_loss = self.mse_loss.forward(input=prediction, target=target)
        phy_loss = self.phy_loss.forward(sample=prediction)

        return self.c1 * mse_loss + self.c2 * phy_loss


class divPK1NeoHookLoss(torch.nn.Module):

    def __init__(self, mu, lmbd, spacing=1.0):

        torch.nn.Module.__init__(self)

        self.lmbd = lmbd
        self.mu = mu
        self.spacing = spacing
        # self.mask_key = mask_key

    def forward(self, sample: torch.Tensor) -> torch.Tensor:

        # put spatial dims first
        pred = rearrange(sample, 'b i j k d -> d i j k b')
        # put spatial dims last to match rearrange done in gradU
        # mask = rearrange(sample[self.mask_key], 'b d i j k -> b i j k d')
        # estimate gradient and hessian with finite differences
        gradU = getGradU3D(pred, self.spacing)
        hessU = getHessU3D(gradU, self.spacing)
        # compute divPK1
        r = divPK1Neohook(gradU, hessU, self.mu, self.lmbd) #* mask
        # remove nan
        r[r != r] = 0
        # summask = np.prod(sample[self.pred_key].shape[:1] + sample[self.pred_key].shape[2:]).item() / max(mask.sum(), 1)
        return r.norm(dim=-1).mean() #* summask


def getGradU3D(field, spacing):
    """
    Computes the gradient of a 3D vector field of shape d i j k b, with d components dim, ijk spatial dims
    and b batch dim. Output b, ijk, component dim (ui,j,k) and spatial deriv (di,j,k).
    """

    gradU = torch.stack(torch.gradient(field, spacing=spacing, dim=[1, 2, 3]))
    # put components dim first (ui,j,k) and then spatial (di,j,k)
    gradU = rearrange(gradU, 'd1 d2 i j k b -> b i j k d2 d1')
    return gradU


def getHessU3D(gradU, spacing):
    """
    Computes the hessian of a 3D gradient of a vector field of shape b i j k d2 d1, with b, batch dim, ijk spatial dims,
    d2 component dim (ui,j,k) and d1 spatial deriv (di,j,k). Output b, ijk, d2, d1, d3 second spatial deriv
    """

    return torch.stack(torch.gradient(gradU, spacing=spacing, dim=[1, 2, 3]), dim=-1)


# this indices matrix helps the computation of cofactor matrices in 3d
_minorIdx3d = torch.empty((3, 3, 2, 2, 2), dtype=int)
for i in torch.arange(3):
    for j in torch.arange(3):
        posi, posj = 0, 0
        for si in torch.arange(3):
            for sj in torch.arange(3):
                if si != i and sj != j:
                    _minorIdx3d[i, j, posi, posj, 0] = si
                    _minorIdx3d[i, j, posi, posj, 1] = sj
                    posj += 1
                    if posj == 2:
                        posi = 1
                        posj = 0
_minorSigns3d = torch.ones((3, 3))
_minorSigns3d[0, 1] = -1
_minorSigns3d[1, 0] = -1
_minorSigns3d[1, 2] = -1
_minorSigns3d[2, 1] = -1

# this indices matrix helps the computation of cofactor matrices in 2d
_minorIdx2d = torch.asarray([
   [[1, 1], [1, 0]],
   [[0, 1], [0, 0]]
])
_minorSigns2d = torch.ones((2, 2))
_minorSigns2d[0, 1] = -1
_minorSigns2d[1, 0] = -1


def _convertTo3D(gradU, hessU=None):
    """
    Converts 2d gradient and hessian matrices to 3d
    """

    dimension = gradU.shape[-1]

    # convert 2d gradient and hessian to 3d
    # using plane strain assumptions (du/dz = 0)
    if dimension == 2:
        dimension = 3
        gradU2d, hessU2d = gradU, hessU
        gradU = torch.zeros((gradU.shape[0], dimension, dimension), device=gradU.device)
        gradU[..., :2, :2] = gradU2d

        if hessU2d is not None:
            hessU = torch.zeros((hessU.shape[0], dimension, dimension, dimension), device=hessU.device)
            hessU[..., :2, :2, :2] = hessU2d

    return gradU, hessU


def divPK1Neohook(gradU, hessU, mu, lmbd):
    """
    Computes the divergence of the first Piola-Kirchhoff tensor of a compressible neohookean material
    """

    # transfer idx matrices to device
    minorIdx2d = _minorIdx2d.to(gradU.device)
    minorIdx3d = _minorIdx3d.to(gradU.device)
    minorSigns2d = _minorSigns2d.to(gradU.device)
    minorSigns3d = _minorSigns3d.to(gradU.device)

    # reshape so that gradU is always N x dim x dim
    # and hessU is always N x dim x dim x dim
    if gradU.ndim < 3: gradU = gradU[None, ...]
    if hessU.ndim < 4: hessU = hessU[None, ...]

    # convert gradients to 3D if necessary
    dimension = 3
    gradU, hessU = _convertTo3D(gradU, hessU)
    # print('dimension', gradU.shape)
    # print('dimension', dimension)

    # compute gradient of deformation
    F = gradU + torch.eye(dimension, device=gradU.device)

    # indexed gradient of deformation [F]i',j'
    # this is the squared matrix with the values of F that would yield to the minors matrix of F
    # if the determinant of this matrix is computed over the last two dimensions
    idxF = F[..., minorIdx3d[..., 0], minorIdx3d[..., 1]]

    # indexed gradient of F (hessian of U)
    # spatial derivatives of gradient of deformation F, indexed as above for F
    # this matrix allows the computation of spatial derivatives with respect to
    # cofactors i,j
    idxHessU = torch.empty((*hessU.shape[0:-3], dimension, dimension, 2, 2, dimension), device=hessU.device)
    for jdx in range(dimension):
        idxHessU[..., jdx] = hessU[..., minorIdx3d[..., 0], minorIdx3d[..., 1], jdx]
    # print('idxHessU computed ...')

    # now compute the cofactor matrices of each squared submatrix from the indexed hessian matrix
    cofIdxHessU = torch.einsum('...pqj,pq->...pqj', idxHessU[..., minorIdx2d[..., 0], minorIdx2d[..., 1], :],
                               minorSigns2d)
    # print('cofIdxHessU computed ...')

    # matrix of cofactors of F
    # compute the determinant of the squared submatrices
    # of the last two dimensions of the indexed F matrix
    # and multiply by minor signs accordingly
    cofF = torch.einsum('...ij,ij->...ij', torch.linalg.det(idxF), minorSigns3d)
    # print('cofF computed ...')
    # the determinant can be computed using any index:
    # det(F) = <F_m1, C_m1> = <F_m2, C_m2> = <F_m3, C_m3>
    # hence its gradient can also be computed using any index.
    # here I use i=1 for both
    i = 0

    # determinant of F and its inverse
    J = torch.einsum('...m,...m', F[..., i], cofF[..., i]) + 1e-8
    invJ = 1 / J
    # print('J and 1 / J computed ...')

    # gradient of cofactor matrix of F
    # NOTE: minorSigns3d = minorSigns3D^T
    gradCofF = torch.einsum('mi,...mipq,...mipqj->...mij', minorSigns3d, idxF, cofIdxHessU)
    # print('gradCofF computed ...')

    # gradient of determinant of F
    gradJ = torch.einsum('...m,...mj->...j', cofF[..., i], hessU[..., i, :])
    gradJ += torch.einsum('...mj,...m->...j', gradCofF[..., i, :], F[..., i])
    # print('gradJ computed ...')

    # divergence of inverse of F transposed
    divInvF_T = -torch.einsum('...,...j,...mj->...m', invJ / J, gradJ, cofF)
    divInvF_T += torch.einsum('...,...mjj->...m', invJ, gradCofF)
    # print('divInvF_T computed ...')

    # divergence of F
    divF = torch.einsum('...mjj->...m', hessU)
    # print('divF computed ...')

    # and finally, divergence of PK1
    divPK1 = mu * (divF - divInvF_T)
    if lmbd != 0.0:
        divPK1 += 0.5 * lmbd * torch.einsum('...,...i->...i', (J ** 2 - 1), divInvF_T)
        divPK1 += lmbd * torch.einsum('...ij,...j->...i', cofF, gradJ)

    return divPK1

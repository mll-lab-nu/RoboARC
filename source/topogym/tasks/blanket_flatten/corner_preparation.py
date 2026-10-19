"""Prepare the approved corner folds with the existing bounded velocity solver."""
import numpy as np

from topogym.world.physics.cloth_preparation import ClothPreparation


class CornerFoldPreparation(ClothPreparation):
    def form(self, origin):
        def target(half, corner):
            p = self.task.preparation_target(self.task.material_uv,
                half_progress=half, corner_progress=corner)
            p[:, 2] -= self.profile.table_clearance_m
            return p+origin
        if self.profile.far_half_fold:
            # Start preparation, offset 15 cm from the arms' home pose and moved back afterwards (2026-10-07).
            away = np.array([0., .15, 0.])
            flat = target(0., 0.)
            self.drive("fold_away", lambda t: flat+t*away, 3.)
            self.drive("half_fold", lambda t: target(t, 0.)+away, 10.)
            self.drive("corner_fold", lambda t: target(1., t)+away, 7.)
            folded = target(1., 1.)+away
            self.drive("fold_return", lambda t: folded-t*away, 4.)
            return
        if self.profile.half_fold:
            self.drive("half_fold", lambda t: target(t, 0.), 10.)
        self.drive("corner_fold", lambda t: target(1., t), 7.)

import vector
import numpy as np

from basis import helicity_basis


def _get_particle_in_frame(particle_dict, frame: str) -> vector.Vector4D:
    if frame not in particle_dict:
        raise ValueError(f"Frame '{frame}' not available.")
    if particle_dict[frame] is None:
        raise ValueError(f"Particle not yet transformed to frame '{frame}'.")
    return particle_dict[frame]

def __mass_preprocess(particle: vector.Vector4D) -> vector.Vector4D:
    """
    Preprocess the mass of the particle
    """
    if particle.m < 1e-3:
        return vector.Vector4D(
            px=particle.px,
            py=particle.py,
            pz=particle.pz,
            m=1e-3,
        )
    return particle
class Core:
    def __init__(
            self,
            main_particle_1: vector.Vector4D, main_particle_2: vector.Vector4D,
            child1: vector.Vector4D, child2: vector.Vector4D, c1_m1=None, c2_m2=None
    ):
        """
        Core class to calculate the cosine distribution between the parent particle and the children particles

        **All particles should be in the LAB Frame**
        :param main_particle_1: particle with positive charge to calculate the spin
        :param main_particle_2: particle with negative charge to calculate the spin
        :param child1: children particle of main_particle_1
        :param child2: children particle of main_particle_2
        """
        self._m1 = {
            'lab frame': main_particle_1,
        }
        self._m2 = {
            'lab frame': main_particle_2,
        }
        self._c1 = {
            'lab frame': child1,
        }
        
        self._c2 = {
            'lab frame': child2,
        }

        self._all_particles = {
            'm1': self._m1,
            'm2': self._m2,
            'c1': self._c1,
            'c2': self._c2,
        }
        self.c1_m1 = c1_m1
        self.c2_m2 = c2_m2

    def m1(self, frame: str) -> vector.Vector4D:
        return _get_particle_in_frame(self._m1, frame)

    def m2(self, frame: str) -> vector.Vector4D:
        return _get_particle_in_frame(self._m2, frame)

    def c1(self, frame: str) -> vector.Vector4D:
        return _get_particle_in_frame(self._c1, frame)

    def c2(self, frame: str) -> vector.Vector4D:
        return _get_particle_in_frame(self._c2, frame)

    def _transform_to_frame(self, boost: vector.Vector3D, source_frame: str, target_frame: str) -> None:
        for particle_dict in self._all_particles.values():
            particle_dict[target_frame] = particle_dict[source_frame].boost(boost)
    

    def analyze(self) -> dict[str, np.ndarray]:
        # Transform to the center of mass frame
        m0 = self.m1('lab frame').add(self.m2('lab frame'))
        self._transform_to_frame(-m0.to_beta3(), 'lab frame', 'cm frame')

        # Transform to the rest frame of the parent particle
        self._transform_to_frame(-self.m1('cm frame').to_beta3(), 'cm frame', 'm1 cm frame')
        self._transform_to_frame(-self.m2('cm frame').to_beta3(), 'cm frame', 'm2 cm frame')
        if self.c1_m1 is not None:
            self._c1['m1 cm frame'] = self.c1_m1
        if self.c2_m2 is not None:
            self._c2['m2 cm frame'] = self.c2_m2

        # Calculate the helicity basis
        helicity_m1 = helicity_basis(self.m1('cm frame'))
        # helicity_m2 = helicity_basis(self.m2('cm frame'))

        # calculate cosine of the angle between the children particles and the helicity basis
        return {
            'theta_tau_cm': 2 * np.arccos(np.abs(self.m1('cm frame').costheta)) / np.pi,
            'cos_theta_A_n': self.c1('m1 cm frame').to_pxpypz().unit().dot(helicity_m1['n']),
            'cos_theta_A_r': self.c1('m1 cm frame').to_pxpypz().unit().dot(helicity_m1['r']),
            'cos_theta_A_k': self.c1('m1 cm frame').to_pxpypz().unit().dot(helicity_m1['k']),
            'c1': self.c1('m1 cm frame'),
            'c2': self.c2('m2 cm frame'),
            'm1CM': self.m1('cm frame'),
            'm2CM': self.m2('cm frame'),

            # 'cos_theta_B_n': self.c2('m2 cm frame').to_pxpypz().unit().dot(helicity_m2['n']),
            # 'cos_theta_B_r': self.c2('m2 cm frame').to_pxpypz().unit().dot(helicity_m2['r']),
            # 'cos_theta_B_k': self.c2('m2 cm frame').to_pxpypz().unit().dot(helicity_m2['k']),

            'cos_theta_B_n': self.c2('m2 cm frame').to_pxpypz().unit().dot(helicity_m1['n']),
            'cos_theta_B_r': self.c2('m2 cm frame').to_pxpypz().unit().dot(helicity_m1['r']),
            'cos_theta_B_k': self.c2('m2 cm frame').to_pxpypz().unit().dot(helicity_m1['k']),
        } 


if __name__ == "__main__":
    m1 = vector.Vector(x=44.9603, y=-0.893463, z=-505.204, t=507.204)
    m2 = vector.Vector(x=-44.9603, y=0.893463, z=-474.914, t=477.041)
    c1 = vector.Vector(x=2.25565, y=0.254229, z=-23.2934, t=23.4042)
    c2 = vector.Vector(x=-14.2044, y=-0.5068570, z=-147.73, t=148.412)

    core = Core(m1, m2, c1, c2)
    print(core.analyze())
import numpy as np


class _KelvinToRGBTable(object):
    """
    refenence augimg
    """
    # Added in 0.4.0.
    def __init__(self):
        self.table = self.create_table()

    def transform_kelvins_to_rgb_multipliers(self, kelvins):
        """Transform kelvin values to corresponding multipliers for RGB images.

        A single returned multiplier denotes the channelwise multipliers
        in the range ``[0.0, 1.0]`` to apply to an image to change its kelvin
        value to the desired one.

        Added in 0.4.0.

        Parameters
        ----------
        kelvins : iterable of number
            Imagewise temperatures in kelvin.

        Returns
        -------
        ndarray
            ``float32 (N, 3) ndarrays``, one per kelvin.

        """
        kelvins = np.clip(kelvins, 1000, 40000)

        tbl_indices = kelvins / 100 - (1000//100)
        tbl_indices_floored = np.floor(tbl_indices)
        tbl_indices_ceiled = np.ceil(tbl_indices)
        interpolation_factors = tbl_indices - tbl_indices_floored

        tbl_indices_floored_int = tbl_indices_floored.astype(np.int32)
        tbl_indices_ceiled_int = tbl_indices_ceiled.astype(np.int32)

        multipliers_floored = self.table[tbl_indices_floored_int, :]
        multipliers_ceiled = self.table[tbl_indices_ceiled_int, :]
        multipliers = (
            multipliers_floored
            + interpolation_factors
            * (multipliers_ceiled - multipliers_floored)
        )

        return multipliers

    # Added in 0.4.0.
    @classmethod
    def create_table(cls):
        table = np.float32([
            [255, 56, 0],  # K=1000
            [255, 71, 0],  # K=1100
            [255, 83, 0],  # K=1200
            [255, 93, 0],  # K=1300
            [255, 101, 0],  # K=1400
            [255, 109, 0],  # K=1500
            [255, 115, 0],  # K=1600
            [255, 121, 0],  # K=1700
            [255, 126, 0],  # K=1800
            [255, 131, 0],  # K=1900
            [255, 137, 18],  # K=2000
            [255, 142, 33],  # K=2100
            [255, 147, 44],  # K=2200
            [255, 152, 54],  # K=2300
            [255, 157, 63],  # K=2400
            [255, 161, 72],  # K=2500
            [255, 165, 79],  # K=2600
            [255, 169, 87],  # K=2700
            [255, 173, 94],  # K=2800
            [255, 177, 101],  # K=2900
            [255, 180, 107],  # K=3000
            [255, 184, 114],  # K=3100
            [255, 187, 120],  # K=3200
            [255, 190, 126],  # K=3300
            [255, 193, 132],  # K=3400
            [255, 196, 137],  # K=3500
            [255, 199, 143],  # K=3600
            [255, 201, 148],  # K=3700
            [255, 204, 153],  # K=3800
            [255, 206, 159],  # K=3900
            [255, 209, 163],  # K=4000
            [255, 211, 168],  # K=4100
            [255, 213, 173],  # K=4200
            [255, 215, 177],  # K=4300
            [255, 217, 182],  # K=4400
            [255, 219, 186],  # K=4500
            [255, 221, 190],  # K=4600
            [255, 223, 194],  # K=4700
            [255, 225, 198],  # K=4800
            [255, 227, 202],  # K=4900
            [255, 228, 206],  # K=5000
            [255, 230, 210],  # K=5100
            [255, 232, 213],  # K=5200
            [255, 233, 217],  # K=5300
            [255, 235, 220],  # K=5400
            [255, 236, 224],  # K=5500
            [255, 238, 227],  # K=5600
            [255, 239, 230],  # K=5700
            [255, 240, 233],  # K=5800
            [255, 242, 236],  # K=5900
            [255, 243, 239],  # K=6000
            [255, 244, 242],  # K=6100
            [255, 245, 245],  # K=6200
            [255, 246, 248],  # K=6300
            [255, 248, 251],  # K=6400
            [255, 249, 253],  # K=6500
            [254, 249, 255],  # K=6600
            [252, 247, 255],  # K=6700
            [249, 246, 255],  # K=6800
            [247, 245, 255],  # K=6900
            [245, 243, 255],  # K=7000
            [243, 242, 255],  # K=7100
            [240, 241, 255],  # K=7200
            [239, 240, 255],  # K=7300
            [237, 239, 255],  # K=7400
            [235, 238, 255],  # K=7500
            [233, 237, 255],  # K=7600
            [231, 236, 255],  # K=7700
            [230, 235, 255],  # K=7800
            [228, 234, 255],  # K=7900
            [227, 233, 255],  # K=8000
            [225, 232, 255],  # K=8100
            [224, 231, 255],  # K=8200
            [222, 230, 255],  # K=8300
            [221, 230, 255],  # K=8400
            [220, 229, 255],  # K=8500
            [218, 228, 255],  # K=8600
            [217, 227, 255],  # K=8700
            [216, 227, 255],  # K=8800
            [215, 226, 255],  # K=8900
            [214, 225, 255],  # K=9000
            [212, 225, 255],  # K=9100
            [211, 224, 255],  # K=9200
            [210, 223, 255],  # K=9300
            [209, 223, 255],  # K=9400
            [208, 222, 255],  # K=9500
            [207, 221, 255],  # K=9600
            [207, 221, 255],  # K=9700
            [206, 220, 255],  # K=9800
            [205, 220, 255],  # K=9900
            [204, 219, 255],  # K=10000
            [203, 219, 255],  # K=10100
            [202, 218, 255],  # K=10200
            [201, 218, 255],  # K=10300
            [201, 217, 255],  # K=10400
            [200, 217, 255],  # K=10500
            [199, 216, 255],  # K=10600
            [199, 216, 255],  # K=10700
            [198, 216, 255],  # K=10800
            [197, 215, 255],  # K=10900
            [196, 215, 255],  # K=11000
            [196, 214, 255],  # K=11100
            [195, 214, 255],  # K=11200
            [195, 214, 255],  # K=11300
            [194, 213, 255],  # K=11400
            [193, 213, 255],  # K=11500
            [193, 212, 255],  # K=11600
            [192, 212, 255],  # K=11700
            [192, 212, 255],  # K=11800
            [191, 211, 255],  # K=11900
            [191, 211, 255],  # K=12000
            [190, 211, 255],  # K=12100
            [190, 210, 255],  # K=12200
            [189, 210, 255],  # K=12300
            [189, 210, 255],  # K=12400
            [188, 210, 255],  # K=12500
            [188, 209, 255],  # K=12600
            [187, 209, 255],  # K=12700
            [187, 209, 255],  # K=12800
            [186, 208, 255],  # K=12900
            [186, 208, 255],  # K=13000
            [185, 208, 255],  # K=13100
            [185, 208, 255],  # K=13200
            [185, 207, 255],  # K=13300
            [184, 207, 255],  # K=13400
            [184, 207, 255],  # K=13500
            [183, 207, 255],  # K=13600
            [183, 206, 255],  # K=13700
            [183, 206, 255],  # K=13800
            [182, 206, 255],  # K=13900
            [182, 206, 255],  # K=14000
            [182, 205, 255],  # K=14100
            [181, 205, 255],  # K=14200
            [181, 205, 255],  # K=14300
            [181, 205, 255],  # K=14400
            [180, 205, 255],  # K=14500
            [180, 204, 255],  # K=14600
            [180, 204, 255],  # K=14700
            [179, 204, 255],  # K=14800
            [179, 204, 255],  # K=14900
            [179, 204, 255],  # K=15000
            [178, 203, 255],  # K=15100
            [178, 203, 255],  # K=15200
            [178, 203, 255],  # K=15300
            [178, 203, 255],  # K=15400
            [177, 203, 255],  # K=15500
            [177, 202, 255],  # K=15600
            [177, 202, 255],  # K=15700
            [177, 202, 255],  # K=15800
            [176, 202, 255],  # K=15900
            [176, 202, 255],  # K=16000
            [176, 202, 255],  # K=16100
            [175, 201, 255],  # K=16200
            [175, 201, 255],  # K=16300
            [175, 201, 255],  # K=16400
            [175, 201, 255],  # K=16500
            [175, 201, 255],  # K=16600
            [174, 201, 255],  # K=16700
            [174, 201, 255],  # K=16800
            [174, 200, 255],  # K=16900
            [174, 200, 255],  # K=17000
            [173, 200, 255],  # K=17100
            [173, 200, 255],  # K=17200
            [173, 200, 255],  # K=17300
            [173, 200, 255],  # K=17400
            [173, 200, 255],  # K=17500
            [172, 199, 255],  # K=17600
            [172, 199, 255],  # K=17700
            [172, 199, 255],  # K=17800
            [172, 199, 255],  # K=17900
            [172, 199, 255],  # K=18000
            [171, 199, 255],  # K=18100
            [171, 199, 255],  # K=18200
            [171, 199, 255],  # K=18300
            [171, 198, 255],  # K=18400
            [171, 198, 255],  # K=18500
            [170, 198, 255],  # K=18600
            [170, 198, 255],  # K=18700
            [170, 198, 255],  # K=18800
            [170, 198, 255],  # K=18900
            [170, 198, 255],  # K=19000
            [170, 198, 255],  # K=19100
            [169, 198, 255],  # K=19200
            [169, 197, 255],  # K=19300
            [169, 197, 255],  # K=19400
            [169, 197, 255],  # K=19500
            [169, 197, 255],  # K=19600
            [169, 197, 255],  # K=19700
            [169, 197, 255],  # K=19800
            [168, 197, 255],  # K=19900
            [168, 197, 255],  # K=20000
            [168, 197, 255],  # K=20100
            [168, 197, 255],  # K=20200
            [168, 196, 255],  # K=20300
            [168, 196, 255],  # K=20400
            [168, 196, 255],  # K=20500
            [167, 196, 255],  # K=20600
            [167, 196, 255],  # K=20700
            [167, 196, 255],  # K=20800
            [167, 196, 255],  # K=20900
            [167, 196, 255],  # K=21000
            [167, 196, 255],  # K=21100
            [167, 196, 255],  # K=21200
            [166, 196, 255],  # K=21300
            [166, 195, 255],  # K=21400
            [166, 195, 255],  # K=21500
            [166, 195, 255],  # K=21600
            [166, 195, 255],  # K=21700
            [166, 195, 255],  # K=21800
            [166, 195, 255],  # K=21900
            [166, 195, 255],  # K=22000
            [165, 195, 255],  # K=22100
            [165, 195, 255],  # K=22200
            [165, 195, 255],  # K=22300
            [165, 195, 255],  # K=22400
            [165, 195, 255],  # K=22500
            [165, 195, 255],  # K=22600
            [165, 194, 255],  # K=22700
            [165, 194, 255],  # K=22800
            [165, 194, 255],  # K=22900
            [164, 194, 255],  # K=23000
            [164, 194, 255],  # K=23100
            [164, 194, 255],  # K=23200
            [164, 194, 255],  # K=23300
            [164, 194, 255],  # K=23400
            [164, 194, 255],  # K=23500
            [164, 194, 255],  # K=23600
            [164, 194, 255],  # K=23700
            [164, 194, 255],  # K=23800
            [164, 194, 255],  # K=23900
            [163, 194, 255],  # K=24000
            [163, 194, 255],  # K=24100
            [163, 193, 255],  # K=24200
            [163, 193, 255],  # K=24300
            [163, 193, 255],  # K=24400
            [163, 193, 255],  # K=24500
            [163, 193, 255],  # K=24600
            [163, 193, 255],  # K=24700
            [163, 193, 255],  # K=24800
            [163, 193, 255],  # K=24900
            [163, 193, 255],  # K=25000
            [162, 193, 255],  # K=25100
            [162, 193, 255],  # K=25200
            [162, 193, 255],  # K=25300
            [162, 193, 255],  # K=25400
            [162, 193, 255],  # K=25500
            [162, 193, 255],  # K=25600
            [162, 193, 255],  # K=25700
            [162, 193, 255],  # K=25800
            [162, 192, 255],  # K=25900
            [162, 192, 255],  # K=26000
            [162, 192, 255],  # K=26100
            [162, 192, 255],  # K=26200
            [162, 192, 255],  # K=26300
            [161, 192, 255],  # K=26400
            [161, 192, 255],  # K=26500
            [161, 192, 255],  # K=26600
            [161, 192, 255],  # K=26700
            [161, 192, 255],  # K=26800
            [161, 192, 255],  # K=26900
            [161, 192, 255],  # K=27000
            [161, 192, 255],  # K=27100
            [161, 192, 255],  # K=27200
            [161, 192, 255],  # K=27300
            [161, 192, 255],  # K=27400
            [161, 192, 255],  # K=27500
            [161, 192, 255],  # K=27600
            [161, 192, 255],  # K=27700
            [160, 192, 255],  # K=27800
            [160, 192, 255],  # K=27900
            [160, 191, 255],  # K=28000
            [160, 191, 255],  # K=28100
            [160, 191, 255],  # K=28200
            [160, 191, 255],  # K=28300
            [160, 191, 255],  # K=28400
            [160, 191, 255],  # K=28500
            [160, 191, 255],  # K=28600
            [160, 191, 255],  # K=28700
            [160, 191, 255],  # K=28800
            [160, 191, 255],  # K=28900
            [160, 191, 255],  # K=29000
            [160, 191, 255],  # K=29100
            [160, 191, 255],  # K=29200
            [159, 191, 255],  # K=29300
            [159, 191, 255],  # K=29400
            [159, 191, 255],  # K=29500
            [159, 191, 255],  # K=29600
            [159, 191, 255],  # K=29700
            [159, 191, 255],  # K=29800
            [159, 191, 255],  # K=29900
            [159, 191, 255],  # K=30000
            [159, 191, 255],  # K=30100
            [159, 191, 255],  # K=30200
            [159, 191, 255],  # K=30300
            [159, 190, 255],  # K=30400
            [159, 190, 255],  # K=30500
            [159, 190, 255],  # K=30600
            [159, 190, 255],  # K=30700
            [159, 190, 255],  # K=30800
            [159, 190, 255],  # K=30900
            [159, 190, 255],  # K=31000
            [158, 190, 255],  # K=31100
            [158, 190, 255],  # K=31200
            [158, 190, 255],  # K=31300
            [158, 190, 255],  # K=31400
            [158, 190, 255],  # K=31500
            [158, 190, 255],  # K=31600
            [158, 190, 255],  # K=31700
            [158, 190, 255],  # K=31800
            [158, 190, 255],  # K=31900
            [158, 190, 255],  # K=32000
            [158, 190, 255],  # K=32100
            [158, 190, 255],  # K=32200
            [158, 190, 255],  # K=32300
            [158, 190, 255],  # K=32400
            [158, 190, 255],  # K=32500
            [158, 190, 255],  # K=32600
            [158, 190, 255],  # K=32700
            [158, 190, 255],  # K=32800
            [158, 190, 255],  # K=32900
            [158, 190, 255],  # K=33000
            [158, 190, 255],  # K=33100
            [157, 190, 255],  # K=33200
            [157, 190, 255],  # K=33300
            [157, 189, 255],  # K=33400
            [157, 189, 255],  # K=33500
            [157, 189, 255],  # K=33600
            [157, 189, 255],  # K=33700
            [157, 189, 255],  # K=33800
            [157, 189, 255],  # K=33900
            [157, 189, 255],  # K=34000
            [157, 189, 255],  # K=34100
            [157, 189, 255],  # K=34200
            [157, 189, 255],  # K=34300
            [157, 189, 255],  # K=34400
            [157, 189, 255],  # K=34500
            [157, 189, 255],  # K=34600
            [157, 189, 255],  # K=34700
            [157, 189, 255],  # K=34800
            [157, 189, 255],  # K=34900
            [157, 189, 255],  # K=35000
            [157, 189, 255],  # K=35100
            [157, 189, 255],  # K=35200
            [157, 189, 255],  # K=35300
            [157, 189, 255],  # K=35400
            [157, 189, 255],  # K=35500
            [156, 189, 255],  # K=35600
            [156, 189, 255],  # K=35700
            [156, 189, 255],  # K=35800
            [156, 189, 255],  # K=35900
            [156, 189, 255],  # K=36000
            [156, 189, 255],  # K=36100
            [156, 189, 255],  # K=36200
            [156, 189, 255],  # K=36300
            [156, 189, 255],  # K=36400
            [156, 189, 255],  # K=36500
            [156, 189, 255],  # K=36600
            [156, 189, 255],  # K=36700
            [156, 189, 255],  # K=36800
            [156, 189, 255],  # K=36900
            [156, 189, 255],  # K=37000
            [156, 189, 255],  # K=37100
            [156, 188, 255],  # K=37200
            [156, 188, 255],  # K=37300
            [156, 188, 255],  # K=37400
            [156, 188, 255],  # K=37500
            [156, 188, 255],  # K=37600
            [156, 188, 255],  # K=37700
            [156, 188, 255],  # K=37800
            [156, 188, 255],  # K=37900
            [156, 188, 255],  # K=38000
            [156, 188, 255],  # K=38100
            [156, 188, 255],  # K=38200
            [156, 188, 255],  # K=38300
            [155, 188, 255],  # K=38400
            [155, 188, 255],  # K=38500
            [155, 188, 255],  # K=38600
            [155, 188, 255],  # K=38700
            [155, 188, 255],  # K=38800
            [155, 188, 255],  # K=38900
            [155, 188, 255],  # K=39000
            [155, 188, 255],  # K=39100
            [155, 188, 255],  # K=39200
            [155, 188, 255],  # K=39300
            [155, 188, 255],  # K=39400
            [155, 188, 255],  # K=39500
            [155, 188, 255],  # K=39600
            [155, 188, 255],  # K=39700
            [155, 188, 255],  # K=39800
            [155, 188, 255],  # K=39900
            [155, 188, 255],  # K=40000
        ]) / 255.0
        _KelvinToRGBTable._TABLE = table
        return table


class ColorTemperature:
    def __init__(self):
        self.table = _KelvinToRGBTable()
        self.kelvins = [2000, 3500, 5500, 7000, 9000, 12000, 20000, 39000]
        return

    def __call__(self, image, factor=None):
        kelvin = factor if factor is not None else np.random.choice(self.kelvins, size=1, replace=False)
        kelvin = np.array(kelvin, dtype=np.float32)
        rgb_multipliers = self.table.transform_kelvins_to_rgb_multipliers(kelvin)
        rgb_multipliers_nhwc = rgb_multipliers.reshape((-1, 1, 3))
        image_temp = np.round(
            image.astype(np.float32) * rgb_multipliers_nhwc
        ).astype(np.uint8)
        return image_temp

    def __str__(self):
        return 'ColorTemperatureOp'


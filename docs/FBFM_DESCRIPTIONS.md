# Behave Fire Behavior Fuel Models

This file lists the defined fuel models available in the vendored `behave_fire` package in this repository, using the numeric model code and alphanumeric ID from `dependencies/behave_fire/src/behave_fire/components/fuel_models.py`.

## Original 13 Fuel Models

| Number | ID | Description |
|---|---|---|
| 1 | FM1 | Short grass [1] |
| 2 | FM2 | Timber grass and understory [2] |
| 3 | FM3 | Tall grass [3] |
| 4 | FM4 | Chaparral [4] |
| 5 | FM5 | Brush [5] |
| 6 | FM6 | Dormant brush, hardwood slash [6] |
| 7 | FM7 | Southern rough [7] |
| 8 | FM8 | Short needle litter [8] |
| 9 | FM9 | Long needle or hardwood litter [9] |
| 10 | FM10 | Timber litter & understory [10] |
| 11 | FM11 | Light logging slash [11] |
| 12 | FM12 | Medium logging slash [12] |
| 13 | FM13 | Heavy logging slash [13] |

## Non-Burnable

| Number | ID | Description |
|---|---|---|
| 91 | NB1 | Urban, developed [91] |
| 92 | NB2 | Snow, ice [92] |
| 93 | NB3 | Agricultural [93] |
| 94 | NB4 | Future standard non-burnable [94] |
| 95 | NB5 | Future standard non-burnable [95] |
| 98 | NB8 | Open water [98] |
| 99 | NB9 | Bare ground [99] |

## Grass

| Number | ID | Description |
|---|---|---|
| 101 | GR1 | Short, sparse, dry climate grass (D) |
| 102 | GR2 | Low load, dry climate grass (D) |
| 103 | GR3 | Low load, very coarse, humid climate grass (D) |
| 104 | GR4 | Moderate load, dry climate grass (D) |
| 105 | GR5 | Low load, humid climate grass (D) |
| 106 | GR6 | Moderate load, humid climate grass (D) |
| 107 | GR7 | High load, dry climate grass (D) |
| 108 | GR8 | High load, very coarse, humid climate grass (D) |
| 109 | GR9 | Very high load, humid climate grass (D) |
| 110 | V-Hb | Short Gass, < 0.5 m (Dynamic) |
| 111 | V-Ha | Tall Grass, > 0.5 m (Dynamic) |

## Grass-Shrub

| Number | ID | Description |
|---|---|---|
| 121 | GS1 | Low load, dry climate grass-shrub (D) |
| 122 | GS2 | Moderate load, dry climate grass-shrub (D) |
| 123 | GS3 | Moderate load, humid climate grass-shrub (D) |
| 124 | GS4 | High load, humid climate grass-shrub (D) |

## Shrub

| Number | ID | Description |
|---|---|---|
| 141 | SH1 | Low load, dry climate shrub (D) |
| 142 | SH2 | Moderate load, dry climate shrub (S) |
| 143 | SH3 | Moderate load, humid climate shrub (S) |
| 144 | SH4 | Low load, humid climate timber-shrub (S) |
| 145 | SH5 | High load, dry climate shrub (S) |
| 146 | SH6 | Low load, humid climate shrub (S) |
| 147 | SH7 | Very high load, dry climate shrub (S) |
| 148 | SH8 | High load, humid climate shrub (S) |
| 149 | SH9 | Very high load, humid climate shrub (D) |
| 150 | SCAL17 | Chamise with Moderate Load Grass, 4 feet (Static) |
| 151 | SCAL15 | Chamise with Low Load Grass, 3 feet (Static) |
| 152 | SCAL16 | North Slope Ceanothus with Moderate Load Grass (Static) |
| 153 | SCAL14 | Manzanita/Scrub Oak with Low Load Grass (Static) |
| 154 | SCAL18 | Coastal Sage/Buckwheat Scrub with Low Load Grass (Static) |
| 155 | V-MH | Short Green Shrub < 1 m With Grass, Discontinuous (< 1 m) often discontinuous and with grass (Dynamic) |
| 156 | V-MMb | Short Shrub < 1 m, Low Dead Fraction and/or Thick Foliage (Static) |
| 157 | V-MAb | Short Shrub < 1 m, High Dead Fraction and/or Thin Fuel (Static) |
| 158 | V-MMa | Tall Shrub > 1 m, Low Dead Fraction and/or Thick Foliage (Static) |
| 159 | V-MAa | Tall Shrub > 1 m, High Dead Fraction and/or Thin Fuel (Static) |

## Timber Understory

| Number | ID | Description |
|---|---|---|
| 161 | TU1 | Light load, dry climate timber-grass-shrub (D) |
| 162 | TU2 | Moderate load, humid climate timber-shrub (S) |
| 163 | TU3 | Moderate load, humid climate timber-grass-shrub (D) |
| 164 | TU4 | Dwarf conifer understory (S) |
| 165 | TU5 | Very high load, dry climate timber-shrub (S) |
| 166 | M-EUCd | Discontinuous Litter Eucalyptus Plantation, With or Without Shrub Understory (Static) |
| 167 | M-H | Deciduous or Conifer Litter, Shrub and Herb Understory |
| 168 | M-F | Deciduous or Conifer Litter, Shrub and Fern Understory (Dynamic) |
| 169 | M-CAD | Deciduous Litter, Shrub Understory (Static) |
| 170 | M-ESC | Sclerophyll Broadleaf Litter, Shrub Understory (Static) |
| 171 | M-PIN | Medium-Long Needle Pine Litter, Shrub Understory (Static) |
| 172 | M-EUC | Eucalyptus Litter, Shrub Understory (Static) |

## Timber Litter

| Number | ID | Description |
|---|---|---|
| 181 | TL1 | Low load, compact conifer litter (S) |
| 182 | TL2 | Low load broadleaf litter (S) |
| 183 | TL3 | Moderate load conifer litter (S) |
| 184 | TL4 | Small downed logs (S) |
| 185 | TL5 | High load conifer litter (S) |
| 186 | TL6 | High load broadleaf litter (S) |
| 187 | TL7 | Large downed logs (S) |
| 188 | TL8 | Long-needle litter (S) |
| 189 | TL9 | Very high load broadleaf litter (S) |
| 190 | F-RAC | Very Compact Litter, Short Needle Conifers (Static) |
| 191 | F-FOL | Compact Litter, Deciduous or Evergreen Foliage (Static) |
| 192 | F-PIN | Litter from Medium-Long Needle Pine Trees (Static) |
| 193 | F-EUC | Pure Eucalyptus Litter, No Understory (Static) |

## Slash and Blowdown

| Number | ID | Description |
|---|---|---|
| 201 | SB1 | Low load activity fuel (S) |
| 202 | SB2 | Moderate load activity or low load blowdown (S) |
| 203 | SB3 | High load activity fuel or moderate load blowdown (S) |
| 204 | SB4 | High load blowdown (S) |

## Notes

- Many additional numeric slots are reserved for future or custom models, but they are not populated by default.
- This list reflects the currently defined entries in the local `behave_fire` source, not a generic external Behave reference.

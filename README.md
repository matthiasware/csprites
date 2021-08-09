# Concept Sprites

## Masks

| Shapes  |  rectangle |  square | circle  |  moon | gelato  | pyramid   |  ellipse |
|---|---|---|---|---|---|---|---|
|  Rotation |![](/data/imgs/angles_rectangle.gif)  | ![](/data/imgs/angles_square.gif)  | ![](/data/imgs/angles_circle.gif)   | ![](/data/imgs/angles_moon.gif)   | ![](/data/imgs/angles_gelato.gif)  |  ![](/data/imgs/angles_pyramid.gif) | ![](/data/imgs/angles_ellipse.gif)  |
|  Scale    | ![](/data/imgs/scale_rectangle.gif)  | ![](/data/imgs/scale_square.gif)  | ![](/data/imgs/scale_circle.gif)   | ![](/data/imgs/scale_moon.gif)   | ![](/data/imgs/scale_gelato.gif)  |  ![](/data/imgs/scale_pyramid.gif) | ![](/data/imgs/scale_ellipse.gif)  |
|  Color    | ![](/data/imgs/colors_rectangle.gif)  | ![](/data/imgs/colors_square.gif)  | ![](/data/imgs/colors_circle.gif)   | ![](/data/imgs/colors_moon.gif)   | ![](/data/imgs/colors_gelato.gif)  |  ![](/data/imgs/colors_pyramid.gif) | ![](/data/imgs/colors_ellipse.gif)  |

## Positions

| Positions | Grid | Shape |
|---|---|---|
| 1^2  | ![](/data/imgs/positions_1.png)  | ![](/data/imgs/positions_1_mask.gif)  |
| 2^2  | ![](/data/imgs/positions_2.png)  | ![](/data/imgs/positions_2_mask.gif)  |
| 4^2  | ![](/data/imgs/positions_4.png)  | ![](/data/imgs/positions_4_mask.gif)  |
| 8^2  | ![](/data/imgs/positions_8.png)  | ![](/data/imgs/positions_8_mask.gif)  |
| 16^2 | ![](/data/imgs/positions_16.png) | ![](/data/imgs/positions_16_mask.gif) |


## Backgrounds

 | Style \ number | single | finite (n=8) | infinite |
 |---|---|---|---|
 | constant          | ![](/data/imgs/bg_constant_color_1.gif) |![](/data/imgs/bg_constant_color_8.gif) | ![](/data/imgs/bg_constant_color_inf.gif) |
 | random pixel      | ![](/data/imgs/bg_random_pixel_1.gif)   |![](/data/imgs/bg_random_pixel_8.gif)   | ![](/data/imgs/bg_random_pixel_inf.gif)   |
 | random structured | ![](/data/imgs/bg_random_function_1.gif)   |![](/data/imgs/bg_random_function_8.gif)   | ![](/data/imgs/bg_random_function_inf.gif)   |
 

# Notes
## Todo
todo:
- Add multi sprites support
- Add randomness via blur and stuff!
- Add lightning conditions
- add final touch function e.g. bluring. shering ect.
- Add classifier
- Self supervised models
- Add sampler when discrete state is too large!
- Allow to specify graphial model for feature distributions and interactions
	discrete vs continuous
	-> sample from the joint distribution

## Features Requests
- Multi-CSprites [min_sprites & max_sprites]
- Overlapping sprites
- allow compositionality
- add more sprites
- build complex sprites from simple sprites
- specify what should be in the target variable
- build website to generate stuff

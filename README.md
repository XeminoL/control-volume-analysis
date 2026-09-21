> [!NOTE]
> **Status:** This project is inactive and no longer updated.

Draw a closed boundary anywhere on the sea. The tool reports how much water
crosses it, and how far you should trust that figure.

![](docs/screenshot.png)

## Two routes, one number

The flux is computed along the boundary you drew, and again from the divergence
of the field enclosed by it. Both should agree. The gap between them is not
hidden: it is reported as part of the answer, alongside the share of that gap
which comes from instrument noise rather than from the method.

## Running it

`run` opens the map in a browser. `run lint` checks the source. 

The bundled field is the Gulf of Tonkin, from [HYCOM](https://www.hycom.org/).
A comparison against ERA5 is available but not required.

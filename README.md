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

`run` opens the map in a browser. `run lint` checks the source. Python 3.11 or
newer; numpy and h5py are installed on first launch.

The bundled field is the Gulf of Tonkin, from [HYCOM](https://www.hycom.org/).
A comparison against ERA5 is available but not required.

## A few deliberate choices

Land is carried as missing data rather than zero, so a coastline never reads as
still water. Area is integrated over the spherical band instead of a flat
rectangle. Instrument error enters through speed and bearing, which is how the
sensor reports it, rather than through the components.

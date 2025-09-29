import pkgutil, gsplat
print([m.name for m in pkgutil.walk_packages(gsplat.__path__, gsplat.__name__ + '.')])

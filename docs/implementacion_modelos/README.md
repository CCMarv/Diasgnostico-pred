# Implementación de modelos

Esta carpeta documenta cómo está implementado cada modelo dentro del proyecto, con foco en la diferencia entre una explicación didáctica y la arquitectura real del repositorio.

## Archivos

- `svm.md`: implementación real de la SVM y sus equivalencias de código.

## Criterio de documentación

Cada archivo debe explicar:

- dónde se define el modelo,
- cómo entra al pipeline,
- qué preprocesamiento comparte,
- cómo se entrena y valida,
- cómo se serializa,
- y cómo termina consumiéndose desde inferencia/API.

La primera guía disponible es la de SVM. Los demás modelos se agregarán siguiendo la misma estructura.

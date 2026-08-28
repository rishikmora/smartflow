{{/* Common labels applied to every object, so `kubectl get all -l app.kubernetes.io/part-of=smartflow` finds the whole stack. */}}
{{- define "smartflow.labels" -}}
app.kubernetes.io/part-of: smartflow
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{/* Fully-qualified image reference, honouring an optional private registry. */}}
{{- define "smartflow.image" -}}
{{- $registry := .root.Values.image.registry -}}
{{- if $registry -}}{{ $registry }}/{{ .repo }}:{{ .root.Values.image.tag }}
{{- else -}}{{ .repo }}:{{ .root.Values.image.tag }}
{{- end -}}
{{- end -}}

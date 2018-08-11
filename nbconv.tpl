{% extends 'script.tpl'%}
{% block any_cell %}
{% if 'export' in cell['metadata'].get('tags', []) %}
   {% if 'first' in cell['metadata'].get('tags', []) %}
cluster = true
       {{ super() }}
   {% else %}
       {{ super() }}
   {% endif %}
{% endif %}
{% endblock any_cell %}

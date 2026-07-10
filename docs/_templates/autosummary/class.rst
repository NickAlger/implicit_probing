{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods %}
   {% set documented_methods = [] %}
   {% for item in methods %}
   {%- if not item.startswith('_') and item not in inherited_members %}
   {{ documented_methods.append(item) or "" }}
   {%- endif %}
   {%- endfor %}
   {% if documented_methods %}
   .. rubric:: {{ _('Methods') }}

   Each method is documented on its own page.

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in documented_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set documented_attributes = [] %}
   {% for item in attributes %}
   {%- if not item.startswith('_') and item not in inherited_members %}
   {{ documented_attributes.append(item) or "" }}
   {%- endif %}
   {%- endfor %}
   {% if documented_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in documented_attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

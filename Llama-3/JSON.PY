import json

data = '```json\n{\n  "Case Importance": "1",\n  "Reasoning": "This case appears to be of high importance because it involves multiple allegations under several articles of the Convention, particularly Article 3, which is central to the prohibition of torture and inhuman or degrading treatment. The case raises significant issues of police conduct, systemic discrimination against Roma, and the adequacy of investigations into allegations of ill-treatment. These issues have the potential to contribute to the development or clarification of the Court\'s case law, especially concerning the procedural obligations under Article 3 and the intersection of discrimination and human rights violations. The case also touches on broader socio-political issues affecting the Roma community, which could have wider implications for the protection of minority rights in Europe."\n}\n```'

json_data = json.loads(data.strip('```json\n').strip('```'))
print(json_data)


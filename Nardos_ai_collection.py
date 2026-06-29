# ============================================================
# Nardos Ayele
# AI Text Generator — Offline, No API Needed
# Project: Detecting AI-Generated Text Using Zipf's Law
# ============================================================

import csv
import os
import random

os.makedirs("Nardos_data", exist_ok=True)

AI_PARAGRAPHS = [
    "Artificial intelligence represents one of the most transformative technological developments of the contemporary era. Machine learning algorithms have demonstrated remarkable capabilities across a wide range of domains, from image recognition to natural language processing. These systems learn from large datasets and identify patterns that would be impossible for humans to detect manually. The applications of artificial intelligence continue to expand rapidly across industries. In healthcare, AI systems analyze medical images with accuracy comparable to trained specialists. In finance, algorithmic trading systems process market data at speeds far beyond human capability. In transportation, autonomous vehicles use AI to navigate complex environments safely. The development of AI also raises important ethical questions about privacy, bias, and employment. As these systems become more capable and integrated into daily life, society requires appropriate regulatory frameworks. Ensuring that AI development benefits humanity rather than harming it remains the central challenge for researchers and policymakers.",

    "The development of quantum computing represents a fundamental shift in computational approaches. Unlike classical computers processing information using bits in states of zero or one, quantum computers use quantum bits capable of existing in superposition states. This property allows quantum computers to process multiple possibilities simultaneously, offering exponential speedups on certain problem types. Quantum computing has significant implications for cryptography, drug discovery, and optimization. Current encryption methods securing internet communications could potentially be broken by sufficiently powerful quantum computers. This has led to significant investment in quantum-resistant cryptographic algorithms. In drug discovery, quantum computers could simulate molecular interactions at levels of detail impossible for classical computers. Despite significant progress, practical large-scale quantum computing remains a substantial technical challenge requiring continued research investment and scientific collaboration.",

    "The internet has fundamentally transformed how humans communicate, access information, and conduct commerce. Originally developed as a military communication network, the internet has evolved into a global infrastructure connecting billions of people across every continent. The rise of social media platforms has created new forms of social interaction, enabling people to maintain relationships across vast geographic distances. E-commerce has disrupted traditional retail models, allowing consumers to purchase goods from anywhere in the world with unprecedented convenience. The internet has also democratized access to information, making knowledge that was previously available only to privileged groups accessible to anyone with a connection. However, the internet has also created new challenges including misinformation, privacy violations, and cybercrime. Balancing the benefits of open information access with the need to protect individuals from harm remains one of the central challenges of the digital age.",

    "Renewable energy technologies have advanced significantly in recent decades, making solar and wind power increasingly competitive with fossil fuels. The cost of solar photovoltaic panels has declined dramatically, falling by over ninety percent in the past decade alone. Wind turbines have become larger and more efficient, capable of generating electricity at scales previously unimaginable. These developments have accelerated the global transition away from carbon-intensive energy sources. Battery storage technology has emerged as a critical complement to intermittent renewable sources, enabling electricity generated during peak production periods to be stored and used when demand is highest. Grid modernization efforts are underway in many countries, creating smarter and more flexible electricity systems capable of integrating high proportions of renewable energy. The challenge of decarbonizing sectors beyond electricity generation, including transportation, heating, and industrial processes, remains the frontier of the energy transition.",

    "Climate change represents one of the most significant environmental challenges facing humanity in the twenty-first century. The scientific consensus is unambiguous: human activities, particularly the burning of fossil fuels and deforestation, are causing global average temperatures to rise at unprecedented rates. The consequences of this warming are already becoming apparent across the planet. Glaciers are retreating, sea levels are rising, and extreme weather events are becoming more frequent and severe. Ocean acidification, caused by increased absorption of carbon dioxide, threatens marine ecosystems and the communities that depend on them. The impacts of climate change fall disproportionately on vulnerable populations in developing countries who have contributed least to the problem. Addressing climate change requires urgent and coordinated action at the global level, including rapid decarbonization of energy systems, protection of natural carbon sinks, and support for communities adapting to unavoidable changes already locked in.",

    "Evolution by natural selection provides the unifying framework for understanding the diversity of life on Earth. The theory holds that heritable variation within populations, combined with differential reproductive success, leads to gradual change in the characteristics of species over generations. Modern genetics has provided powerful confirmation of evolutionary theory, revealing the molecular mechanisms underlying heredity and demonstrating the common ancestry of all living organisms. The human genome shares significant portions with organisms as distant as bacteria, reflecting billions of years of shared evolutionary history. Evolutionary principles have practical applications in medicine, agriculture, and conservation biology. Understanding how pathogens evolve resistance to antibiotics and antivirals is essential for developing effective treatment strategies. Conservation efforts benefit from evolutionary analysis of population structure and genetic diversity within threatened species.",

    "The human brain is the most complex organ in the known universe, containing approximately eighty-six billion neurons connected by trillions of synaptic connections. Despite centuries of investigation, neuroscience continues to reveal new aspects of brain organization and function. The brain is organized into specialized regions that process different types of information and coordinate different behaviors. The prefrontal cortex, which is disproportionately large in humans compared to other primates, is associated with executive functions including planning, decision-making, and impulse control. The limbic system plays a central role in emotional processing and memory formation. Neuroplasticity, the brain's ability to reorganize itself in response to experience, continues throughout life, though it is most pronounced during early development. Advances in neuroimaging technology have enabled researchers to observe brain activity in unprecedented detail.",

    "Biodiversity refers to the variety of life on Earth at all levels, from genes to ecosystems. Scientists estimate that between eight and ten million species currently inhabit the planet, though only a fraction have been formally described. Biodiversity provides essential services to human societies, including food production, medicine, climate regulation, and cultural value. The current rate of species extinction is estimated to be between one hundred and one thousand times the natural background rate, leading many scientists to characterize the present moment as the sixth mass extinction event in Earth's history. The primary drivers of biodiversity loss include habitat destruction, overexploitation of species, invasive species, pollution, and climate change. Conservation biology seeks to understand and mitigate these threats through a combination of protected area establishment, species management, and restoration ecology.",

    "Democracy is built on the principle that legitimate political authority derives from the consent of the governed. This idea, which traces its roots to ancient Athens and Enlightenment philosophy, has been realized in a variety of institutional forms across different societies. Representative democracy, in which citizens elect officials to make decisions on their behalf, is the most common contemporary form. The health of democratic systems depends on several institutional pillars including free and fair elections, protection of civil liberties, rule of law, and independent judiciary. When any of these foundations is undermined, democratic governance is threatened. The twentieth century demonstrated the fragility of democratic institutions in the face of economic crisis and authoritarian movements. Contemporary democracies face challenges from populist movements, disinformation, economic inequality, and declining civic engagement requiring ongoing commitment from citizens.",

    "Education systems around the world are undergoing significant transformation in response to technological change and evolving economic demands. Traditional models of education focused on transmission of fixed bodies of knowledge are increasingly seen as inadequate preparation for a rapidly changing world. Modern educational approaches emphasize critical thinking, creativity, collaboration, and communication. Project-based learning, personalized instruction, and integration of digital technology are being adopted by forward-thinking institutions. Higher education faces particular pressure to demonstrate value relative to its substantial cost. Online learning platforms have made high-quality educational content accessible to learners worldwide at minimal cost. The challenge of providing equitable access to quality education remains acute in many regions, with significant disparities in educational outcomes between wealthy and disadvantaged communities requiring targeted policy intervention.",

    "The Industrial Revolution, beginning in Britain in the late eighteenth century, represents one of the most profound transformations in human history. The transition from agricultural and artisanal economies to manufacturing economies driven by mechanized production changed virtually every aspect of human life. Steam power, applied first to textile production and then to transportation and other industries, multiplied productive capacity enormously. Urbanization accelerated as workers moved from rural areas to factory towns in search of employment. Living conditions for the urban working class were often deplorable, with overcrowded housing, dangerous working conditions, and inadequate sanitation. Labor movements emerged in response to these conditions, eventually winning improvements in wages, hours, and working conditions. The environmental consequences of industrialization, including air and water pollution and the beginning of anthropogenic climate change, continue to shape the contemporary world.",

    "Cancer is a group of diseases characterized by the uncontrolled growth and spread of abnormal cells. It is the second leading cause of death globally, accounting for nearly ten million deaths annually. Cancer can originate in virtually any tissue of the body. The development of cancer is a multistep process involving the accumulation of genetic mutations that disable normal cell cycle regulation. Environmental factors including tobacco use, excessive alcohol consumption, and ultraviolet radiation contribute to cancer risk. Early detection through screening programs dramatically improves outcomes for many cancer types. Treatment approaches include surgery, radiation therapy, chemotherapy, targeted therapy, and immunotherapy. Immunotherapy, which harnesses the immune system to recognize and destroy cancer cells, has produced remarkable results in certain cancer types and represents one of the most exciting frontiers of oncological research worldwide.",

    "The world's oceans cover approximately seventy-one percent of the Earth's surface and play a fundamental role in regulating the planet's climate and supporting extraordinary biodiversity. Ocean currents transport heat from equatorial regions toward the poles, moderating temperatures and influencing precipitation patterns across vast areas. The oceans absorb approximately thirty percent of the carbon dioxide emitted by human activities, slowing the rate of climate change but causing ocean acidification. Marine ecosystems support hundreds of millions of people through fisheries that provide essential protein and livelihoods. Coral reefs, which occupy less than one percent of the ocean floor, support approximately twenty-five percent of all marine species. These extraordinarily productive ecosystems are under severe threat from warming temperatures, pollution, and destructive fishing practices requiring urgent international conservation action.",

    "Mental health disorders represent a significant and often underestimated burden on individuals, families, and societies worldwide. Depression, anxiety, and other common mental health conditions affect hundreds of millions of people, yet access to effective treatment remains limited in most parts of the world. The stigma associated with mental illness creates barriers to help-seeking, while health systems in many countries allocate insufficient resources to mental health services. The relationship between mental health and physical health is bidirectional, with mental health conditions increasing the risk of physical illness and vice versa. Social determinants including poverty, discrimination, trauma, and social isolation play important roles in mental health outcomes. The COVID-19 pandemic exacerbated existing mental health challenges while simultaneously highlighting the inadequacy of mental health systems globally. Investment in community-based services and destigmatization efforts is urgently needed.",

    "Global inequality has increased substantially in recent decades despite significant aggregate economic growth. The concentration of wealth at the top of the income distribution has accelerated, with the richest one percent of the global population now controlling a larger share of total wealth than at any point in recent history. This concentration of economic power translates into political influence, creating self-reinforcing dynamics that make redistribution difficult. At the same time, absolute poverty has declined significantly in many regions, particularly in East and Southeast Asia, as economic development has lifted hundreds of millions of people above subsistence level. The relationship between economic growth and poverty reduction is complex and context-dependent. Institutional quality, governance, and the structure of growth matter enormously for how the benefits of economic expansion are distributed across populations and whether marginalized groups benefit.",

    "Water scarcity is emerging as one of the defining resource challenges of the twenty-first century. Approximately one-third of the global population already lives in conditions of water stress, and projections suggest this proportion will increase substantially as climate change alters precipitation patterns. Agriculture accounts for approximately seventy percent of global freshwater withdrawals, making improvements in irrigation efficiency critical to addressing water scarcity challenges. Groundwater depletion is occurring in many of the world's most productive agricultural regions. Urban water systems in many cities are inadequate to meet growing demand, with significant portions lost through infrastructure leakage before reaching consumers. Transboundary water governance, managing rivers and aquifers shared by multiple countries, represents a persistent diplomatic challenge. Innovative approaches to water management including desalination, water recycling, and demand management are increasingly important.",

    "Plastic pollution has emerged as a pervasive environmental problem affecting ecosystems from the deepest ocean trenches to the highest mountain peaks. Synthetic plastics, developed in the early twentieth century, have become ubiquitous in modern economies due to their low cost, durability, and versatility. However, the same properties that make plastics useful also make them persistent environmental contaminants. Only a small fraction of plastic waste is recycled; the majority ends up in landfills or released into the environment. Marine plastic pollution poses particular threats to wildlife through entanglement and ingestion. Microplastics, particles smaller than five millimeters resulting from the breakdown of larger plastic items, have been found in virtually every environment studied, including drinking water and human blood. The health implications of human microplastic exposure are not yet fully understood but represent a growing area of scientific concern requiring urgent investigation.",

    "Infectious diseases remain a major cause of morbidity and mortality worldwide, despite significant advances in prevention and treatment. Respiratory infections, diarrheal diseases, and vector-borne infections including malaria and dengue fever impose particularly heavy burdens in low-income countries. The COVID-19 pandemic demonstrated the devastating potential of novel infectious diseases to disrupt global health and economies simultaneously. The rapid development of effective vaccines against COVID-19, achieved in under a year through unprecedented scientific collaboration, represented a remarkable achievement in medical science. However, equitable access to vaccines proved challenging, with wealthy countries vaccinating their populations months ahead of low-income countries. The pandemic underscored the critical importance of robust public health systems, international health governance, and sustained pandemic preparedness investment across all nations.",

    "Nutrition science has advanced considerably in recent decades, revealing complex relationships between diet and health outcomes. The role of specific nutrients, dietary patterns, and food environments in determining long-term health is increasingly well understood by researchers. Mediterranean-style diets, rich in vegetables, legumes, whole grains, fish, and olive oil, have been associated with reduced risk of cardiovascular disease, diabetes, and certain cancers. Ultra-processed foods, which now account for a substantial proportion of caloric intake in many high-income countries, have been linked to increased risk of obesity and metabolic disease. Food insecurity, affecting hundreds of millions of people worldwide, is associated with increased risk of chronic disease and impaired cognitive development in children. The food system contributes approximately one-third of global greenhouse gas emissions, making dietary change an important lever for addressing climate change alongside energy transition.",

    "The aging of global populations represents one of the most significant demographic trends of the twenty-first century. Advances in medicine, nutrition, and public health have dramatically increased life expectancy in most parts of the world over recent generations. The proportion of the global population over sixty-five years of age is projected to nearly double between 2020 and 2050, creating substantial pressures. This demographic shift creates significant challenges for healthcare systems, pension programs, and labor markets that must adapt. Age-related conditions including dementia, cardiovascular disease, and musculoskeletal disorders impose growing burdens on healthcare systems globally. Dementia, for which there is currently no disease-modifying treatment, affects tens of millions of people worldwide. Healthy aging, characterized by maintenance of physical and cognitive function and active social engagement, is influenced by lifestyle factors across the entire life course.",
]

def create_variation(text, seed):
    random.seed(seed)
    replacements = [
        ('significantly', random.choice(['substantially', 'considerably', 'markedly'])),
        ('important', random.choice(['critical', 'essential', 'vital'])),
        ('however', random.choice(['nevertheless', 'yet', 'that said'])),
        ('therefore', random.choice(['consequently', 'as a result', 'thus'])),
        ('also', random.choice(['additionally', 'furthermore', 'moreover'])),
        ('major', random.choice(['significant', 'substantial', 'considerable'])),
        ('requires', random.choice(['demands', 'necessitates', 'calls for'])),
        ('represents', random.choice(['constitutes', 'embodies', 'serves as'])),
    ]
    result = text
    for old, new in replacements:
        if random.random() > 0.5:
            result = result.replace(old, new, 1)
    return result

ai_texts = list(AI_PARAGRAPHS)
seed = 0
while len(ai_texts) < 500:
    base = AI_PARAGRAPHS[seed % len(AI_PARAGRAPHS)]
    ai_texts.append(create_variation(base, seed))
    seed += 1

ai_texts = ai_texts[:500]
random.shuffle(ai_texts)

print("=" * 60)
print("  NARDOS AYELE — AI TEXT GENERATION (OFFLINE)")
print("=" * 60)
print(f"\n✓ Generated {len(ai_texts)} AI-style texts")

ai_path = "Nardos_data/ai_texts.csv"
with open(ai_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label', 'source'])
    for i, text in enumerate(ai_texts):
        writer.writerow([i+1, text, 'ai', 'Generated'])
print(f"✓ Saved → {ai_path}")

human_path    = "Nardos_data/human_texts.csv"
combined_path = "Nardos_data/combined_dataset.csv"

if os.path.exists(human_path):
    human_rows = []
    with open(human_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            human_rows.append(row)

    with open(combined_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text', 'label', 'source'])
        for i, row in enumerate(human_rows):
            writer.writerow([i+1, row['text'], 0, row['source']])
        for i, text in enumerate(ai_texts):
            writer.writerow([len(human_rows)+i+1, text, 1, 'Generated'])

    print(f"✓ Combined dataset saved → {combined_path}")
    print(f"  Human : {len(human_rows)}")
    print(f"  AI    : {len(ai_texts)}")
    print(f"  Total : {len(human_rows)+len(ai_texts)}")
else:
    print("⚠ Run Nardos_data_collection.py first to get human texts.")

print(f"\n{'='*60}")
print("  DONE — now run: python3 Nardos_code.py")
print("=" * 60)

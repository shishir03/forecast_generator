import ollama

def simplify_discussion(discussion_text):
    extraction_response = ollama.chat(
        model='llama3.1:8b-instruct-q4_K_M',
        messages=[
            {
                'role': 'system',
                'content': """Extract every meteorologically significant claim 
                from the following forecast discussion as a bullet list. 
                Quote directly from the text where possible, and do not add any 
                information not present in the text. Only include the bullet list
                in your response."""
            },
            {
                'role': 'user',
                'content': discussion_text
            }
        ]
    )
    extracted_claims = extraction_response['message']['content']
    print(f"{extracted_claims}\n")

    response = ollama.chat(
        model='llama3.1:8b-instruct-q4_K_M',
        messages=[
            {
                'role': 'system',
                'content': """You are a meteorologist providing a weather forecast 
                for a general audience. Translate the following meteorological claims into 
                plain language for a general audience, providing a single summary for the 
                entire forecast period. Do not add any information beyond what is listed.
                
                Your output must follow this exact format:
                PATTERN: 2-3 sentences describing the large-scale synoptic weather pattern
                IMPACTS: 4-5 sentences describing what this means for local weather
                CONFIDENCE: Low, medium, or high
                """
            },
            {
                'role': 'user',
                'content': f"""Translate these claims:\n\n{extracted_claims}. 
                Only include the simplified text in your response."""
            }
        ]
    )

    return response['message']['content']

print(simplify_discussion(
""".SHORT TERM...
Issued at 1201 PM PDT Sat Apr 18 2026
(This evening through Sunday)

High clouds continue to stream in from the west, yet will have very
little impact on high temperatures this afternoon. We are
forecasting mid 60s to lower 70s in the northwest facing locations
and low-to-upper 70s elsewhere. There is a greater than 50%
probability for Concord, San Jose, Gilroy, Hollister, and King City
to exceed 80 degrees F on this afternoon (but less than 10% of
exceeding 85 degrees F).

Tonight and into Sunday morning, expecting low clouds to return to
the coast and coastal adjacent valleys as moisture increases ahead
of an approaching mid/upper level low. This low will also cool
temperatures slightly as clouds increase, most notability in the
North Bay and San Francisco Bay Area. Meanwhile, the Central Coast
will remain quite warm across the interior Sunday afternoon.

.LONG TERM...
Issued at 1201 PM PDT Sat Apr 18 2026
(Sunday night through next Friday)

The progression of the anticipated cold front has slowed down by
about 12-18 hours. However, we still expect pre-frontal rain showers
to begin Monday morning across the North Bay and then spread
southward across the Bay Area. Outside of the coastal ranges of the
Central Coast, we may see very little rainfall on during the day
Monday. The main cold front is now expected to move across the Bay
Area and Central Coast on Tuesday morning. By Tuesday afternoon and
evening, we have the greatest potential for thunderstorms with up to
30% across much of the region as up to 500 J/kg of surface based
CAPE is forecast. We are mostly forecasting this rainfall to be
beneficial, but urban and poorly drained areas may experience
flooding concerns during periods of heavy rain showers and/or
thunderstorms. The WPC has a Marginal Risk of Excessive Rainfall (at
least 5%) from the Santa Cruz Mountains northward on Day 3 (5 AM
Monday - 5 AM Tuesday). However, we are not expecting any major
river flooding with this event.

Post-frontal rain showers and possible thunderstorms (generally less
than 15%) linger into Wednesday afternoon as a colder air mass
settles in behind the cold front. Drier conditions look to return to
the region for the latter half of the week with troughing forecast
by the clusters."""
))

import requests
from rdflib import RDF, Graph, Literal, URIRef
from rdflib.namespace import FOAF, XSD


def load_rdf_from_url(url, format="xml"):
    # Load RDF data from a given URL.
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        msg = f"failed to load RDF data from {url}"
        raise Exception(msg)


# Create a Graph with SQLite store
uri = Literal("sqlite://")
graph = Graph("SQLAlchemy", identifier="http://example.com/mygraph")

# Open the store
graph.open(uri, create=True)

graph.parse(
    data=load_rdf_from_url("https://homosaurus.org/v3.jsonld"),
    format="json-ld",
)

# url_ = "https://raw.githubusercontent.com/Superraptor/GSSO/master/releases/1.0.0/gsso_v1.0.0_rdf_xml.owl"
# graph.parse(
#     data=load_rdf_from_url(url_),
#     format="xml",
# )

# Create a URIRef for Bob
bob = URIRef("http://example.org/people/Bob")

# Add triples to the graph
graph.add((bob, RDF.type, FOAF.Person))
graph.add((bob, FOAF.name, Literal("Bob", datatype=XSD.string)))
graph.add((bob, FOAF.age, Literal(42, datatype=XSD.integer)))

# Query the graph
print("Querying the graph:")
for s, p, o in graph:
    print(f"{s} {p} {o}")

# Add more data
for i in range(1000):
    person = URIRef(f"http://example.org/people/Person{i}")
    graph.add((person, RDF.type, FOAF.Person))
    graph.add((person, FOAF.name, Literal(f"Person {i}", datatype=XSD.string)))
    graph.add((person, FOAF.age, Literal(i % 100, datatype=XSD.integer)))

print(f"\nTotal number of triples: {len(graph)}")

# Perform a SPARQL query
query = """
SELECT ?name ?age
WHERE {
    ?person rdf:type foaf:Person ;
            foaf:name ?name ;
            foaf:age ?age .
    FILTER (?age > 30 && ?age < 40)
}
LIMIT 5
"""

print("\nQuerying for people between 30 and 40 years old:")
for row in graph.query(query):
    print(f"Name: {row.name}, Age: {row.age}")

# Remember to close the graph when you're done
graph.close()

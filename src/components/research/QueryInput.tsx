import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Search, Sparkles } from "lucide-react";

interface QueryInputProps {
  onSubmit: (query: string) => void;
  isLoading?: boolean;
}

export const QueryInput = ({ onSubmit, isLoading = false }: QueryInputProps) => {
  const [query, setQuery] = useState("");

  const exampleQueries = [
    "Summarize the three latest papers on LLM safety, recommend one open-source dataset mentioned in them, and plot the publication trend in this topic over the past year.",
    "Find recent research on transformer architectures for multimodal AI and analyze the performance improvements over the past 6 months.",
    "Research the current state of federated learning in healthcare applications and identify key challenges mentioned in recent publications."
  ];

  const handleSubmit = () => {
    if (query.trim() && !isLoading) {
      onSubmit(query.trim());
    }
  };

  return (
    <Card className="p-6 bg-gradient-card shadow-card">
      <div className="space-y-4">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="h-5 w-5 text-primary" />
          <h2 className="text-lg font-semibold text-foreground">Research Query</h2>
        </div>
        
        <Textarea
          placeholder="Ask me to research, analyze, and synthesize information from multiple sources..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="min-h-[120px] resize-none border-border focus:ring-primary"
          disabled={isLoading}
        />
        
        <div className="flex justify-between items-center">
          <div className="text-sm text-muted-foreground">
            {query.length}/1000 characters
          </div>
          <Button 
            onClick={handleSubmit}
            disabled={!query.trim() || isLoading}
            variant="research"
            size="lg"
          >
            <Search className="h-4 w-4 mr-2" />
            {isLoading ? "Researching..." : "Start Research"}
          </Button>
        </div>
      </div>

      <div className="mt-6">
        <h3 className="text-sm font-medium text-muted-foreground mb-3">Example Queries</h3>
        <div className="space-y-2">
          {exampleQueries.map((example, index) => (
            <Button
              key={index}
              variant="ghost"
              size="sm"
              className="w-full text-left justify-start h-auto p-3 text-sm text-muted-foreground hover:text-foreground hover:bg-accent"
              onClick={() => setQuery(example)}
              disabled={isLoading}
            >
              "{example}"
            </Button>
          ))}
        </div>
      </div>
    </Card>
  );
};
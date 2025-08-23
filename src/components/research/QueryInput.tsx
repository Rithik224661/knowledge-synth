import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Search, Sparkles, Copy } from "lucide-react";

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

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSubmit();
    }
  };

  return (
    <div className="space-y-6">
      <Card className="p-8 card-elevated">
        <div className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-8 h-8 bg-gradient-research rounded-lg flex items-center justify-center">
              <Sparkles className="h-4 w-4 text-primary-foreground" />
            </div>
            <h2 className="text-2xl font-semibold text-foreground">Research Query</h2>
          </div>
          
          <Textarea
            placeholder="Ask me to research, analyze, and synthesize information from multiple sources..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyPress}
            className="min-h-[140px] resize-none border-border/60 focus:ring-primary/20 focus:border-primary text-base leading-relaxed"
            disabled={isLoading}
          />
          
          <div className="flex justify-between items-center">
            <div className="text-sm text-muted-foreground">
              {query.length}/1000 characters • Press Ctrl+Enter to submit
            </div>
            <Button 
              onClick={handleSubmit}
              disabled={!query.trim() || isLoading}
              variant="research"
              size="lg"
              className="btn-research px-6 py-3"
            >
              <Search className="h-5 w-5 mr-2" />
              {isLoading ? "Researching..." : "Start Research"}
            </Button>
          </div>
        </div>
      </Card>

      <Card className="p-6 card-elevated">
        <h3 className="text-lg font-semibold text-foreground mb-4">Example Queries</h3>
        <div className="space-y-3">
          {exampleQueries.map((example, index) => (
            <div
              key={index}
              className="example-query group"
              onClick={() => setQuery(example)}
            >
              <div className="flex items-start justify-between">
                <p className="text-sm text-muted-foreground leading-relaxed pr-4">
                  "{example}"
                </p>
                <Button
                  variant="ghost"
                  size="sm"
                  className="opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                  onClick={(e) => {
                    e.stopPropagation();
                    setQuery(example);
                  }}
                  disabled={isLoading}
                >
                  <Copy className="h-3 w-3" />
                </Button>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};
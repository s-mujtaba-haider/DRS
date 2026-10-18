import { Suspense } from "react";

import MatchControl from "@/components/MatchControl";
import { Empty } from "@/components/ui";

async function Loader({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <MatchControl id={id} />;
}

export default function Page({ params }: { params: Promise<{ id: string }> }) {
  return (
    <Suspense fallback={<Empty>Loading match...</Empty>}>
      <Loader params={params} />
    </Suspense>
  );
}
